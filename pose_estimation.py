import cv2
import numpy as np
import pandas as pd
import os

proto = "C:\Sports_Pose_Assignment\pose_deploy_linevec.prototxt"
model = "C:\Sports_Pose_Assignment\pose_iter_440000.caffemodel"
print("Proto exists:", os.path.exists(proto))
print("Model exists:", os.path.exists(model))
net = cv2.dnn.readNetFromCaffe(proto, model)
print("Model loaded successfully")

BODY_PARTS = {
    0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist",
    8: "RHip", 9: "RKnee", 10: "RAnkle",
    11: "LHip", 12: "LKnee", 13: "LAnkle"
}

PAIRS = [
    (1,2),(2,3),(3,4),
    (1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),
    (1,11),(11,12),(12,13)
]

cap = cv2.VideoCapture("SampleSideonSideBatting.mp4")
print("Video opened:", cap.isOpened())

ret, frame = cap.read()
print("First frame read:", ret)


os.makedirs("output", exist_ok=True)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "output/pose_overlay.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

frame_id = 0
data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(
        frame, 1.0/255, (368, 368),
        (0, 0, 0), swapRB=False, crop=False
    )
    net.setInput(blob)
    output = net.forward()

    points = []

    for i in range(len(BODY_PARTS)):
        heatmap = output[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatmap)
        x = int((frame.shape[1] * point[0]) / output.shape[3])
        y = int((frame.shape[0] * point[1]) / output.shape[2])

        if conf > 0.1:
            points.append((x, y))
            data.append([frame_id, i, x, y])
        else:
            points.append(None)

    for pair in PAIRS:
        partA, partB = pair
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0,255,0), 2)
            cv2.circle(frame, points[partA], 4, (0,0,255), -1)

    out.write(frame)
    frame_id += 1

cap.release()
out.release()

df = pd.DataFrame(data, columns=["frame", "keypoint", "x", "y"])
df.to_csv("output/keypoints.csv", index=False)

print("Pose estimation completed (Python 3.14 compatible)")
