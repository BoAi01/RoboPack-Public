import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import cv2
import re
import argparse
from tqdm import tqdm
from utils.visualizer import play_and_save_video

# Function to extract numeric values from a filename
def extract_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return -1

def merge_videos(video_directory, output_filename, num_videos=None, show_progress=False, fps=30, every_n_vid=1, repeat_last_frame=0):
    # Get a list of video filenames sorted by their numeric values
    video_files = os.listdir(video_directory)
    video_files = list(filter(lambda x: x.endswith('.mp4'), video_files))
    video_files = sorted(video_files, key=lambda x: extract_number(x))

    if not video_files:
        print("No video files found in the directory.")
        return

    # Determine frame size based on the first video
    first_video_path = os.path.join(video_directory, video_files[0])
    cap = cv2.VideoCapture(first_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    cap.release()

    # Limit the number of videos to merge
    if num_videos is None:
        num_videos = len(video_files)

    # Initialize a list to store frames
    frames = []

    # Loop through the sorted video files and extract frames
    for idx, video_file in enumerate(tqdm(video_files[:num_videos], disable=not show_progress, desc="Extracting Frames")):
        
        # Process frames based on the every_n_vid argument
        if idx % every_n_vid != 0:
            continue
    
        video_path = os.path.join(video_directory, video_file)
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)            
            
        cap.release()

        # Repeat the last frame for the specified duration
        if repeat_last_frame > 0:
            last_frame = frames[-1]
            repeat_frames = [last_frame] * int(fps * repeat_last_frame)
            frames.extend(repeat_frames)

    # Use the play_and_save_video function to save the frames as a video
    play_and_save_video(frames, output_filename, fps)

    print(f"Merged video saved as {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge videos in a directory into a single video.")
    parser.add_argument("--vid-dir", type=str, help="Directory containing video clips")
    parser.add_argument("--output-path", type=str, help="Output merged video filename")
    parser.add_argument("--num-videos", type=int, default=None, help="Number of videos to merge (default: all)")
    parser.add_argument("--show-progress", action="store_true", help="Display progress bar")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (FPS) of the generated video (default: 30)")
    parser.add_argument("--every-n-vid", type=int, default=1, help="Choose a video every n frames")
    parser.add_argument("--repeat-last-frame", type=int, default=0, help="Repeat the last frame for n seconds")

    args = parser.parse_args()
    merge_videos(args.vid_dir, args.output_path, args.num_videos, args.show_progress, args.fps, args.every_n_vid, args.repeat_last_frame)
