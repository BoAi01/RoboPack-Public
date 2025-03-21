import numpy as np
import pdb


class Sampler:
    def __init__(self):
        pass


class GaussianSampler(Sampler):
    def __init__(self, horizon, a_dim):
        self.horizon = horizon
        self.a_dim = a_dim

    def sample_actions(self, num_samples, mu, std):
        return (
            np.expand_dims(mu, 0)
            + np.random.normal(size=(self.num_samples, self.a_dim)) * std[0]
        )


class CorrelatedNoiseSampler(GaussianSampler):
    def __init__(self, a_dim, beta, horizon, num_repeat=1):
        # Beta is the correlation coefficient between each timestep
        # the smaller the the beta, the smaller the correlation 
        super().__init__(horizon, a_dim)
        self.beta = beta
        self.num_repeat = num_repeat
        assert horizon % num_repeat == 0, f"horizon {horizon} cannot be divided by num_repeat {num_repeat}"

    def sample_actions(self, num_samples, mu, std):
        noise_samples = [np.random.normal(size=(num_samples, self.a_dim)) * std[0]] * self.num_repeat

        while len(noise_samples) < self.horizon:
            # a sample is the sum of an uncorrelated part and a
            # correlated part, weighted by self.beta
            noise_samp = (
                self.beta * noise_samples[-1]
                + (1 - self.beta)
                * np.random.normal(size=(num_samples, self.a_dim))
                * std[len(noise_samples)]
            )
            # repeat the samples for self.num_repeat times
            for _ in range(self.num_repeat):
                noise_samples.append(noise_samp)

        noise_samples = np.stack(noise_samples, axis=1)
        # noise_samples[:] = mu
        # noise_samples[:, :] = [0, 0.2, 0]
        
        return np.expand_dims(mu, 0) + noise_samples


class StraightLinePushSampler(object):
    """
    A sampler that returns starting-ending positions for end effector motion.
    """

    def __init__(self, horizon, push_distance, action_size=0.008):
        self.horizon = horizon
        self.push_distance = push_distance
        self.action_size = action_size
        self.num_actions = np.ceil(push_distance / action_size)

    def sample_actions(self, num_samples, box_particles, particle_weights, theta_mu, theta_std, alpha_mu, alpha_std):
        starting_ending_pairs, theta_samples, contact_particle_indices, lead_portions = [], [], [], []
        for i in range(num_samples):
            good_sample = False
            while not good_sample:
                # choose contact particle
                contact_particle_idx = np.random.choice(len(particle_weights), p=particle_weights)
                contact_particle = box_particles[contact_particle_idx].cpu().numpy()

                # choose theta
                # if i == 0:
                #     theta = np.zeros((self.horizon, 1)) - np.pi/2
                # else:
                #     theta = np.random.normal(theta_mu, theta_std, (self.horizon, 1))
                theta = np.random.normal(theta_mu, theta_std, (self.horizon, 1))

                # choose push lead portion
                if alpha_mu is None:
                    # lead_portion = np.random.random(1) * 0.6 + 0.4
                    lead_portion = np.random.random(1)  # we may no longer need to hard-code desirable values for lead_portion, given it is statistically estimated now
                else:
                    lead_portion = np.random.normal(alpha_mu, alpha_std, 1)

                done = False
                while not done:
                    distance_before_contact = self.push_distance * lead_portion
                    distance_after_contact = self.push_distance * (1 - lead_portion)

                    # compute the starting and ending end-effector positions
                    starting_position = contact_particle[:2] - np.concatenate([np.cos(theta) * distance_before_contact, np.sin(theta) * distance_before_contact], axis=1)
                    ending_position = contact_particle[:2] + np.concatenate([np.cos(theta) * distance_after_contact, np.sin(theta) * distance_after_contact], axis=1)

                    # if the distance between starting_position and any box point is less than 5cm, increase the lead portion
                    if np.linalg.norm(starting_position - box_particles.cpu().numpy()[:, :2], axis=-1).min() < 0.05:
                        lead_portion += 0.05
                    else:
                        done = True

                if np.bitwise_and(starting_position[:, 0] > 0.675, starting_position[:, 1] < -0.1).any() or (starting_position[:, 0] > 0.8).any(): # don't go OOB to the right
                    continue
                # Append the starting and ending positions to the list
                lead_portions.append(lead_portion)
                starting_ending_pairs.append((starting_position, ending_position))
                theta_samples.append(theta)
                contact_particle_indices.append(contact_particle_idx)

                good_sample = True
        # Convert starting and ending positions to actions
        starting_positions, actions = self.convert_ee_poses_to_actions(starting_ending_pairs)
        return starting_positions, actions, theta_samples, contact_particle_indices, lead_portions

    def convert_ee_poses_to_actions(self, pairs):
        starting_positions = np.array([pair[0] for pair in pairs])
        ending_positions = np.array([pair[1] for pair in pairs])
        long_actions = ending_positions - starting_positions

        # break long actions into smaller ones
        unit_action = long_actions / self.num_actions   #  np.linalg.norm(long_actions, axis=-1)[: , None]
        short_actions = np.repeat(unit_action, self.num_actions, axis=1) 
        
        return starting_positions, short_actions


class PackingStraightLinePushSampler(object):
    """
    A sampler that returns starting-ending positions for end effector motion for the packing task.
    
    It evaluates two candidate actions, pushing the first row and pushing the second row
    The push is from right to left (along the -y direction)
    The location of the rows can be determined by the point cloud of the scene, thus some visual observations will be required, though
        it does not need to be accurate, since it is only used to gauge the rough position of the objects 
        
    We asssume the robot has grasped the in-hand object and is ready for any action. It means the robot could be waiting for action to execute, or in
        the mist of executing previous actions (e.g., pushing one of the rows)
    """

    def __init__(self, num_actions, action_size=0.0025, allzero_allowed=True):
        self.action_size = action_size
        self.num_actions = num_actions
        # self.pushing_z = pushing_z
        self.NUM_POINTS_PER_OBJ = 20
        self.NUM_ROWS = 2
        self.OBJ_RADIUS = 0.045   # 10cm as object radius
        self.allzero_allowed = allzero_allowed
        
    def get_contact_points(self, table_object_pc): 
        # assume the input is (k, N, 3) where k is the num of objects, and N is the num of particles
        
        # each object has self.NUM_POINTS_PER_OBJ points
        # get the center of each object
        objects = table_object_pc
        object_centers = [pc.mean(0) for pc in objects]
        
        # identify the right-most objects, the ones with the largest y value?
        sorted_object_centers = sorted(object_centers, key=lambda x: x[1].item(), reverse=True)  # from largest to smallest
        
        # use their centers as an approximation of the real contact points
        contacts = sorted_object_centers[:self.NUM_ROWS]
        
        # sort the contact points by their x-axis value so that it is more deterministic
        contacts = sorted(contacts, key=lambda x: x[0])
        
        return contacts

    def get_relocate_ee_action(self, contact_point, curr_pose_xy):
        # if it is already pushing, then no need to relocate ee
        # identify if pushing is already happening by looking at the diff between contact and curr pose
        
        # if the curr x is close to contact point'x, and it their y's are also close or the curr y is smaller than contact point's y
        # then it means a pushing is happening
        is_pushing = abs(contact_point[0] - curr_pose_xy[0]) < self.OBJ_RADIUS and \
                curr_pose_xy[1] < contact_point[1] + 3 * self.OBJ_RADIUS
        # print(curr_pose_xy, contact_point)
        # print(f'is pushing = {is_pushing}', abs(contact_point[0] - curr_pose_xy[0]) < self.OBJ_RADIUS, curr_pose_xy[1] < contact_point[1] + 2 * self.OBJ_RADIUS)
        
        if is_pushing:
            # the relocation action is zero 
            return np.zeros(3)
        else:
            # relocate EE to the other row 
            relocate_ee_pos = np.zeros(2)
            relocate_ee_pos[0] = contact_point[0]       # same x valuea as contact point 
            relocate_ee_pos[1] = contact_point[1] + 3 * self.OBJ_RADIUS
            assert 0.2 < relocate_ee_pos[0] < 0.8 and -0.4 < relocate_ee_pos[1] < 0.4,  \
                f"the ee position to relocate to seems out of the workspace? {relocate_ee_pos}"
            
            # compute diff
            action = np.zeros(3)   # z displacement is assumed to be zero
            action[:2] = relocate_ee_pos - curr_pose_xy 
        
            return action  

    def sample_actions(self, num_samples, table_object_pc, curr_pose):
        assert num_samples % self.NUM_ROWS == 0, f"num_samples {num_samples} should be a multiple of num of rows {self.NUM_ROWS}"
        
        # we asssume the box is aligned with the y axis, and the action should go from +y to -y
        # first, we identify the two contact points, corresponding to the two rows
        contact_points = self.get_contact_points(table_object_pc)    # a list of (3,)
        assert len(contact_points) == self.NUM_ROWS, "num of contact points should equal to the number of rows"
        
        # sample actions for pushing each of the contact points
        action_seqs = []
        for row_idx in range(self.NUM_ROWS): 
            contact_point = contact_points[row_idx]   # contact_points[row_idx]
            
            for _ in range(num_samples // self.NUM_ROWS):
            
                # locate EE to the right pose
                relocate_action = self.get_relocate_ee_action(contact_point, curr_pose)
                
                # get pushing actions
                if self.allzero_allowed:
                    pushing_actions = [np.array([0, -self.action_size, 0]) for i in range(np.random.randint(low=(self.num_actions + 1) //2, high=self.num_actions + 1))]
                else:
                    pushing_actions = [np.array([0, -self.action_size, 0]) for i in range(np.random.randint(low=3*(self.num_actions+1)//4, high=self.num_actions + 1))]
                if len(pushing_actions) < self.num_actions:
                    pushing_actions += [np.zeros(3)] * (self.num_actions - len(pushing_actions))
                    
                action_seqs.append([relocate_action] + pushing_actions)
                
        return np.stack(action_seqs, axis=0)
