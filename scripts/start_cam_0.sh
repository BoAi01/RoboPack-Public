#!/bin/bash

SERIAL_NUMBER=${SERIAL_NUMBER:-"125322061326"}              # Connect to the camera with serial number
EXIT_IF_CAMERA_NOT_FOUND=${EXIT_IF_CAMERA_NOT_FOUND:-true} # true: exit if not found; false: keep running if not found
CAMERA_NAME=${CAMERA_NAME:-"cam_0"}                      # Set the ROS2 topic name
USB_PORT=${USB_PORT:-4-3}         # USB port, found in realsense-viewer
# /sys/devices/pci0000:00/0000:00:1c.4/0000:06:00.0/usb4/4-3/4-3:1.0/video4linux/video6

die (){
  echo "$@" 1>&2
  exit 1

}

if rs-enumerate-devices -s | grep $SERIAL_NUMBER;  then
  echo "Found camera"
else
  echo -e "\e[31mCamera with serial number $SERIAL_NUMBER not found"
  if [ "$EXIT_IF_CAMERA_NOT_FOUND" = true ]; then
    die "Check your camera serial number or set EXIT_IF_CAMERA_NOT_FOUND to false"
#  else
#    echo "Connecting to any camera..."
  fi
fi

# To differentiate between cameras on different USB ports, you can add a parameter:
# e.g. usb_port_id:=2-2.4
CAMERA_PARAMS="camera_name:=$CAMERA_NAME serial_number:=$SERIAL_NUMBER usb_port_id:=$USB_PORT"
GENERAL_PARAMS="log_level:=WARN align_depth.enable:=false"  # align_depth.enable:=true doesn't work
#DEPTH_PARAMS="color_module.profile:=1280x720x15 depth_module.profile:=1280x720x15 depth_module.enable_auto_exposure.1:=false filters:=temporal pointcloud.enable:=true"
DEPTH_PARAMS="color_module.profile:=1280x720x15 depth_module.profile:=1280x720x15 depth_module.enable_auto_exposure.1:=false filters:=temporal pointcloud.enable:=true"

echo "Using parameters: " $CAMERA_PARAMS $GENERAL_PARAMS $DEPTH_PARAMS
# Some warnings are logged by rs_launch.py about parameters not in the correct range (unrelated to the ones in this script)
ros2 launch realsense2_camera rs_launch.py $CAMERA_PARAMS $GENERAL_PARAMS $DEPTH_PARAMS
