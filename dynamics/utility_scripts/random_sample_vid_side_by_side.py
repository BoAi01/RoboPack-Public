import os
import random
import argparse
import numpy as np
import cv2
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from PIL import Image
from utils.visualizer import play_and_save_video


# Define your play_videos_side_by_side function here if it's not already implemented
def play_videos_side_by_side(video_paths, titles, pause_time_last_frame, path=None):
    # Check if videos were opened successfully
    videos = [cv2.VideoCapture(path) for path in video_paths]
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

    for i in range(len(titles)):
        cv2.putText(title_frames[i], titles[i], text_origins[i], font, font_scale, font_color, 2, cv2.LINE_AA)

    frames = []
    while True:
        # Read frames from all videos
        ret_frames = [video.read() for video in videos]

        # Break the loop if any video has ended
        if not all([is_available for is_available, _ in ret_frames]):
            break

        # Resize frames to the same width
        frames_resized = [cv2.resize(frame, (width, height)) for _, frame in ret_frames]
        
        # Stack frames and titles side by side
        top_row = cv2.hconcat(title_frames)
        bottom_row = cv2.hconcat(frames_resized)
        combined_frame = cv2.vconcat([top_row, bottom_row])
        frames.append(combined_frame)

        # Display the combined frame
        # cv2.imshow('Combined Video', combined_frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    # # Pause for a longer time on the last frame
    # cv2.waitKey(pause_time_last_frame)

    # Release video objects and close OpenCV windows
    for video in videos:
        video.release()
    cv2.destroyAllWindows()

    to_save = 'y' #  input(f'Save video to {path}? (y/n)')
    if to_save == 'y':
        play_and_save_video(frames, path, 2)
        print('saved to', path)


def random_sample_and_concatenate_videos(source_dir, target_dir, n, k, pause_time_last_frame):
    video_files = [os.path.join(source_dir, filename) for filename in os.listdir(source_dir) if filename.endswith(".mp4")]

    for i in range(k):
        # Randomly sample n videos
        selected_videos = random.sample(video_files, n)
        video_titles = [os.path.basename(video) for video in selected_videos]
        output_video_path = os.path.join(target_dir, f"output_video_{i}.mp4")

        # Call the existing function to concatenate and save the videos
        play_videos_side_by_side(selected_videos, video_titles, pause_time_last_frame=pause_time_last_frame, path=output_video_path)

def main():
    parser = argparse.ArgumentParser(description="Randomly sample and concatenate videos")
    parser.add_argument("--source-dir", type=str, help="Source directory containing video files")
    parser.add_argument("--target-dir", type=str, help="Target directory to save output videos")
    parser.add_argument("--n", type=int, help="Number of videos to sample at a time")
    parser.add_argument("--k", type=int, help="Number of times to repeat the process")
    parser.add_argument("--pause_time_last_frame", type=int, default=5, help="Pause time in seconds for the last frame")

    args = parser.parse_args()

    # Create the target directory if it doesn't exist
    os.makedirs(args.target_dir, exist_ok=True)

    random_sample_and_concatenate_videos(args.source_dir, args.target_dir, args.n, args.k, args.pause_time_last_frame)

if __name__ == "__main__":
    main()
