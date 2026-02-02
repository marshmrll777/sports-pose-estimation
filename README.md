## Sports Analytics – Human Pose Estimation (Cricket Batting)

## Objective
The objective of this assignment is to analyze a cricket batting video using computer vision techniques and extract human pose keypoints for sports analytics.

## Approach
- OpenCV’s DNN module was used for human pose estimation.
- A pre-trained Caffe-based OpenPose model (COCO body format) was applied.
- Each frame of the input video was processed to detect body joints.
- Detected joints were drawn as a skeleton overlay on the video.
- Joint coordinates were saved in CSV format for analysis.

## Input
- Side-view cricket batting video in MP4 format.

## Output
- pose_overlay.mp4  
  Video showing the detected human pose skeleton overlaid on the original video.

- keypoints.csv  
  CSV file containing body joint coordinates (x, y) and confidence values for each frame.

## How to Run
1. Install required libraries:
   pip install opencv-python numpy pandas

2. Place the input video in the input folder.

3. Run the script:
    pose_estimation.py

4. Output files will be generated in the output folder.

## Notes
This project focuses on pose extraction and sports analytics rather than UI or UX design.

## Note: 
The pose model file (pose_iter_440000.caffemodel) is not included due to GitHub size limits and should be downloaded separately.
