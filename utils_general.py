import os
import h5py
import numpy
import numpy as np
import yaml


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def read_extrinsics(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    extrinsics = {}
    for cam, info in data.items():
        extrinsics[cam] = np.array(info['transformation'])

    return extrinsics


height_diff = 0.15


def get_grasp_pos(object_center):
    gripper_z = max(object_center[-1] + height_diff, 0)
    gripper_pos = object_center.copy()
    gripper_pos[-1] = gripper_z
    return gripper_pos


def create_sequential_folder(directory, prefix="seq_", create=True):
    """
    Returns a formatted folder name following the expression seq_x
    :param directory: the directory under which the folder is
    :return: the folder path
    """
    # Get the list of folders in the directory
    folder_names = os.listdir(directory)

    # Filter out non-folder entries and extract the sequential numbers
    seq_numbers = []
    for folder_name in folder_names:
        if os.path.isdir(os.path.join(directory, folder_name)) and folder_name.startswith(prefix):
            seq_numbers.append(int(folder_name[len(prefix):]))

    # Find the maximum sequential number
    if seq_numbers:
        max_seq_number = max(seq_numbers)
    else:
        max_seq_number = 0

    # Create the new folder with the next sequential number
    new_folder_name = f"{prefix}{max_seq_number + 1}"
    new_folder_path = os.path.join(directory, new_folder_name)
    if create:
        os.makedirs(new_folder_path)

    return new_folder_path


def save_dictionary_to_hdf5(data, file_path):
    """
    Saves a dictionary to an HDF5 file.

    Args:
        data (dict): The dictionary to be saved.
        file_path (str): The file path where the dictionary will be saved.
    """
    with h5py.File(file_path, 'w') as file:
        # Save each key-value pair as a dataset in the file
        for key, value in data.items():
            file.create_dataset(key, data=value)


def load_dictionary_from_hdf5(file_path):
    """
    Loads a dictionary from an HDF5 file.

    Args:
        file_path (str): The file path of the HDF5 file.

    Returns:
        dict: The loaded dictionary.
    """
    dictionary = {}
    with h5py.File(file_path, 'r') as file:
        # Load each dataset and populate the dictionary
        for key in file.keys():
            value = file[key][()]
            dictionary[key] = value
    return dictionary


def find_max_seq_folder(directory):
    max_x = -1
    max_folder_path = None

    for folder in os.listdir(directory):
        if folder.startswith('seq_'):
            x = folder.split('_')[-1]
            try:
                x = int(x)
                if x > max_x:
                    max_x = x
                    max_folder_path = os.path.join(directory, folder)
            except ValueError:
                pass

    return max_folder_path


def get_top_elements(lst, func, N):
    """
    Returns the N elements from the list with the maximum numerical values
    based on the output of the provided function.

    Args:
        lst (list): The input list.
        func (function): A function that takes an element from the list as input and returns a numerical value.
        N (int): The number of elements to return.

    Returns:
        list: The N elements with the maximum numerical values.
    """
    # Sort the list based on the output of the provided function
    sorted_lst = sorted(lst, key=func, reverse=True)
    # Return the top N elements
    return sorted_lst[:N]


def get_directory_contents(directory):
    """
    Returns a list of all subdirectories and file paths within a directory.

    Args:
        directory (str): Directory path.

    Returns:
        list: List of subdirectories and file paths.
    """
    contents = []

    if os.path.isdir(directory):
        for root, dirs, files in os.walk(directory):
            for directory_name in dirs:
                subdirectory = os.path.join(root, directory_name)
                contents.append(subdirectory)
            for file_name in files:
                file_path = os.path.join(root, file_name)
                contents.append(file_path)
    else:
        contents.append(directory)

    return contents


def break_trajectory_dic_into_sequence(trajectory_dic):
    """
    Breaks a dictionary where every value is a trajectory (first dim is T) into a sequence
    of dictionaries where each one is the data for a particular time step.

    :param trajectory_dic: a dictionary of trajectory data
    :return: a sequence of dictionaries
    """
    # first check the T for all value fields are the same
    T = None
    for key, value in trajectory_dic.items():
        if T is None:
            T = value.shape[0]
        else:
            assert value.shape[0] == T, \
                f'all value fields in a trajectory should have the same T, but {value.shape[0]} != {T} for {key}'

    t_dics = []
    for t in range(T):
        single_step_dic = {}
        for key, trajecoty in trajectory_dic.items():
            data_at_t = trajecoty[t]
            single_step_dic[key] = data_at_t
        t_dics.append(single_step_dic)

    return t_dics


def replace_consecutive_failing_elements(lst, boolean_func, num_consecutive, replacement_value=None):
    result = lst.copy()
    consecutive_count = 0

    for i in range(len(result)):
        if result[i] is None:
            continue

        if boolean_func(result[i]):
            consecutive_count = 0
        else:
            consecutive_count += 1
            if consecutive_count >= num_consecutive:
                for j in range(i - consecutive_count + 1, i + 1):
                    result[j] = replacement_value  # Replace with the desired value

    return result


class Queue(object):
    def __init__(self, max_size, init_list=[]):
        self.max_size = max_size
        self.queue = init_list
        assert isinstance(init_list, list)

    def add(self, item):
        if len(self.queue) < self.max_size:
            self.queue.append(item)
        else:
            obj = self.pop()
            del obj
            self.queue.append(item)

    def pop(self):
        if self.queue:
            return self.queue.pop(0)
        else:
            raise ValueError("Queue is empty")

    def __iter__(self):
        return iter(self.queue)

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        return str(self.queue)


def merge_same_consecutive_arrays(arrays):
    merged = []
    arrays = arrays.copy()      # to avoid in-place modification
    for i in range(len(arrays)):
        if i == 0:
            # If it's the first element, simply append it to the merged list.
            merged.append(arrays[i])
        elif np.abs(arrays[i] - arrays[i - 1]).max() < 1e-8:   # avoid numerical error
            # If the current array is the same as the previous one, skip it.
            merged[-1] += arrays[i]
        else:
            # If they are different, append the current array to the merged list.
            merged.append(arrays[i])
    return merged
