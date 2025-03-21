import os
import numpy as np
import cv2
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from PIL import Image
from utils.visualizer import play_and_save_video


# def play_video_side_by_side(video1, video2, title1, title2, pause_time_last_frame, path=None):
#     # Check if videos were opened successfully
#     if not video1.isOpened() or not video2.isOpened():
#         print("Error opening video files")
#         exit()

#     # Get the dimensions of the video frames
#     width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Create title frames
#     title_height = height // 5
#     title1_frame = np.ones((title_height, width, 3), dtype=np.uint8) * 255  # White background
#     title2_frame = np.ones((title_height, width, 3), dtype=np.uint8) * 255  # White background

#     # Configure font properties
#     font_scale = 2
#     font_color = (0, 0, 0)  # Black
#     font = cv2.FONT_HERSHEY_DUPLEX

#     # Calculate text size and position for titles
#     text_size1 = cv2.getTextSize(title1, font, font_scale, 2)[0]
#     text_size2 = cv2.getTextSize(title2, font, font_scale, 2)[0]

#     text_origin1 = ((width - text_size1[0]) // 2, (title_height + text_size1[1]) // 2)
#     text_origin2 = ((width - text_size2[0]) // 2, (title_height + text_size2[1]) // 2)

#     cv2.putText(title1_frame, title1, text_origin1, font, font_scale, font_color, 2, cv2.LINE_AA)
#     cv2.putText(title2_frame, title2, text_origin2, font, font_scale, font_color, 2, cv2.LINE_AA)

#     frames = []
#     while True:
#         # Read frames from both videos
#         ret1, frame1 = video1.read()
#         ret2, frame2 = video2.read()

#         # Break the loop if either video has ended
#         if not ret1 or not ret2:
#             break

#         # Resize frames to the same width
#         frame2_resized = cv2.resize(frame2, (width, height))

#         # Stack frames and titles in two rows
#         top_row = cv2.hconcat([title1_frame, title2_frame])
#         bottom_row = cv2.hconcat([frame1, frame2_resized])
#         combined_frame = cv2.vconcat([top_row, bottom_row])

#         frames.append(combined_frame)

#         # Display the combined frame
#         cv2.imshow('Combined Video', combined_frame)

#         # Exit loop if 'q' is pressed
#         if cv2.waitKey(500) & 0xFF == ord('q'):
#             break

#     # Pause for a longer time on the last frame
#     # cv2.imshow('Combined Video', combined_frame)
#     # cv2.waitKey(pause_time_last_frame)

#     # Release video objects and close OpenCV windows
#     video1.release()
#     video2.release()
#     cv2.destroyAllWindows()

#     to_save = input(f'Save video to {path}? (y/n)')
#     if to_save == 'y':
#         from utils.visualizer import play_and_save_video
#         play_and_save_video(frames, path, 2)
#         print('saved')


# def play_videos_side_by_side(video_paths, titles, pause_time_last_frame, path=None):
#     # Check if videos were opened successfully
#     videos = [cv2.VideoCapture(path) for path in video_paths]
#     for video in videos:
#         if not video.isOpened():
#             print("Error opening video files")
#             exit()

#     # Get the dimensions of the video frames
#     width = int(videos[0].get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(videos[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Create title frames
#     title_height = height // 5
#     title_frames = [np.ones((title_height, width, 3), dtype=np.uint8) * 255 for _ in range(len(titles))]  # White background

#     # Configure font properties
#     font_scale = 1
#     font_color = (0, 0, 0)  # Black
#     font = cv2.FONT_HERSHEY_DUPLEX

#     # Calculate text size and position for titles
#     text_sizes = [cv2.getTextSize(title, font, font_scale, 2)[0] for title in titles]
#     text_origins = [((width - size[0]) // 2, (title_height + size[1]) // 2) for size in text_sizes]

#     for i in range(len(titles)):
#         cv2.putText(title_frames[i], titles[i], text_origins[i], font, font_scale, font_color, 2, cv2.LINE_AA)

#     frames = []
#     while True:
#         # Read frames from all videos
#         ret_frames = [video.read() for video in videos]

#         # Break the loop if any video has ended
#         if not all([is_available for is_available, _ in ret_frames]):
#             break

#         # Resize frames to the same width
#         frames_resized = [cv2.resize(frame, (width, height)) for _, frame in ret_frames]
        
#         # Stack frames and titles side by side
#         top_row = cv2.hconcat(title_frames)
#         bottom_row = cv2.hconcat(frames_resized)
#         combined_frame = cv2.vconcat([top_row, bottom_row])
#         frames.append(combined_frame)

#         # Display the combined frame
#         # cv2.imshow('Combined Video', combined_frame)

#         # Exit loop if 'q' is pressed
#         if cv2.waitKey(500) & 0xFF == ord('q'):
#             break

#     # # Pause for a longer time on the last frame
#     # cv2.waitKey(pause_time_last_frame)

#     # Release video objects and close OpenCV windows
#     for video in videos:
#         video.release()
#     cv2.destroyAllWindows()

#     to_save = 'y' #  input(f'Save video to {path}? (y/n)')
#     if to_save == 'y':
#         play_and_save_video(frames, path, 2)
#         print('saved to', path)


import numpy as np
import cv2

def play_videos_stacked_vertically(video_paths, titles, pause_time_last_frame, path=None):
    # Check if videos were opened successfully
    videos = [cv2.VideoCapture(video_path) for video_path in video_paths]
    for video in videos:
        if not video.isOpened():
            print("Error opening video files")
            exit()

    # Get the dimensions of the video frames
    width = int(videos[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videos[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create title frames
    title_height = height // 5
    title_frames = [np.ones((title_height, width, 3), dtype=np.uint8) * 255 for _ in range(len(titles))]  # White background

    # Configure font properties
    font_scale = 1
    font_color = (0, 0, 0)  # Black
    font = cv2.FONT_HERSHEY_DUPLEX

    # Calculate text size and position for titles
    text_sizes = [cv2.getTextSize(title, font, font_scale, 2)[0] for title in titles]
    text_origins = [((width - size[0]) // 2, (title_height + size[1]) // 2) for size in text_sizes]

    for i, title_frame in enumerate(title_frames):
        cv2.putText(title_frame, titles[i], text_origins[i], font, font_scale, font_color, 2, cv2.LINE_AA)

    frames = []
    while True:
        # Read frames from all videos
        ret_frames = [video.read() for video in videos]

        # Break the loop if any video has ended
        if not all([is_available for is_available, _ in ret_frames]):
            break

        # Resize frames to the same width
        frames_resized = [cv2.resize(frame, (width, height)) for _, frame in ret_frames]

        # Stack frames and titles vertically
        combined_frames = [cv2.vconcat([title_frames[i], frames_resized[i]]) for i in range(len(videos))]
        combined_frame = cv2.hconcat(combined_frames)
        frames.append(combined_frame)

        # Display the combined frame
        # cv2.imshow('Combined Video', combined_frame)

        # # Exit loop if 'q' is pressed
        # if cv2.waitKey(500) & 0xFF == ord('q'):
        #     break

    # # Pause for a longer time on the last frame
    # cv2.waitKey(pause_time_last_frame)

    # # Release video objects and close OpenCV windows
    # for video in videos:
    #     video.release()
    # cv2.destroyAllWindows()

    to_save = 'y' #  input(f'Save video to {path}? (y/n)')
    if to_save == 'y':
        play_and_save_video(frames, path, 2)
        print('saved to', path)



# root1 = '/home/albert/Downloads/version10_slim'
# root2 = '/home/albert/Downloads/version11_slim'
# save_dir = 'training/train_tb_logs/v24_estimator_predictor/10_3_box7'
save_dir = "/svl/u/boai/robopack/dynamics/training/train_tb_logs/v2_0127_packing_v1_anno/5_6_multiview_seq60"
os.makedirs(save_dir, exist_ok=True)

root_dir = '/svl/u/boai/robopack/dynamics/training/train_tb_logs/v2_0127_packing_v1_anno'
paths = [
    os.path.join(root_dir, 'version_5', 'visualizations_reproduce_figvis_packing_multiview'),
    os.path.join(root_dir, 'version_6', 'visualizations_reproduce_figvis_packing_multiview')
]

titles = [
    # '10 (tac)',
    # '3 (notac)'
    '5 (tac)',
    '6 (notac)'
]

# 17/18, 44, 66, 95, 119, 149
# vid_indices = [17, 18, 19, 44, 45, 46, 66, 67, 68, 94, 95, 118, 119, 120, 148, 149, 150]        # for test set. remember to set the interval to 4
# vid_indices = [17, 18, 19, 39, 40, 41, 42, 59, 60, 61, 86, 87, 107, 108, 109, 132, 133]        # for test set. remember to set the interval to 4
# vid_indices = [8, 17, 26, 34, 46, 58, 73, 84, 94, 105, 120, 129, 143, 146, 147, 168, 169, 183, 197, 211, 224, 237, 251, 264, 280, 292]
# vid_indices = [18, 19, 20, 21, 42, 43, 44,]
# vid_indices = [15, 16, 17, 18, 39, 40, 41, 57, 58, 59, 60, 86, 87, 88, 89, 107, 108, 109, 135, 136, 137]        # for test set. remember to set the interval to 4
# vid_indices = [137, 138, 161, 162, 163, 185, 186, 187, 209, 210, 228, 229, 251, 252]        # for test set. remember to set the interval to 4
# vid_indices = [23, 24, 50, 51, 75, 76, 98, 99, 125, 126]
# vid_indices = [304, 302, 303, 301, 299, 310, 300, 289, 44, 297]

vid_indices = list(range(700))

def main():
    for file in sorted(os.listdir(paths[1])):
        if not file.endswith('.mp4'):
            continue

        # skip unimportant videos 
        index_in_name = False
        for index in vid_indices:
            # if index == int(file.split('_')[2][:3]):
            if index == int(file.split('_')[1][:-4]):
                index_in_name = True
                break
        
        # assert paths exist
        # import pdb; pdb.set_trace()
        # for path in paths:
        #     assert os.path.exists(os.path.join(path, file)), f'{file} not found in {path}'

        # generate combined video 
        if index_in_name:
            for path in paths:
                assert os.path.exists(os.path.join(path, file)), f'{file} not found in {path}'
            print(f'playing {file}')
            video_paths = [os.path.join(path, file) for path in paths]
            play_videos_stacked_vertically(video_paths, titles, 3000,
                                        os.path.join(save_dir, f'{file[:-4]}_combined.mp4'))

        # if file.endswith('.mp4') and os.path.exists(os.path.join(root2, file)):
        #     print(f'playing {file}')
        #     video1 = cv2.VideoCapture(os.path.join(root1, file))
        #     video2 = cv2.VideoCapture(os.path.join(root2, file))
        #     play_video_side_by_side(video2, video1, 'baseline', 'ours', 3000,
        #                             os.path.join(save_dir, f'{file[:-4]}_combined.mp4'))


if __name__ == '__main__':
    main()
