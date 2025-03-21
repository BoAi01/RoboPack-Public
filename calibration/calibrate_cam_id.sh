#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Please provide the ID as an argument."
  exit 1
fi

# Store the ID from the argument
ID1="$1"

echo "Executing calibration for ID: $ID1"

# Execute the command
roslaunch apriltag_ros continuous_detection.launch camera_name:=/$ID1/color image_topic:=image_raw camera_frame:=${ID1}_color_optical_frame
