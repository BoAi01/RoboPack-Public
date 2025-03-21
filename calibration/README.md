# Multi-View Camera Calibration

This repo provides a pipeline for calibrating multiple RGB-D cameras. It largely follows [this guide](https://gist.github.com/hshi74/edabc1e9bed6ea988a2abd1308e1cc96) with three main steps:

1. **Initial calibration** using AprilTags to estimate relative camera poses, using common off-the-shelf libraries.
2. **Manual refinement** by visualizing the fused point cloud and adjusting extrinsics.
3. **Frame alignment** between the robot and camera frames.

> **Note**: Step 2 (manual refinement) is the most important for accurate results. 

### Usage

To calibrate or visualize the setup:

```bash
roslaunch handtune_extrinsics.launch
python publish_joint_state.py
python handtune_control.py
