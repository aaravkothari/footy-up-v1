# from ultralytics import YOLO

# model = YOLO("models\\best.pt")

# results = model.predict(source="input_videos/youtube1.mp4", device=0, save=True)
# print(results[0])
# print("==================")
# for box in results[0].boxes:
#     print(box)

from ultralytics import YOLO
import cv2
import numpy as np
import os

VIDEO_PATH = "input_videos/youtube1.mp4"
OUTPUT_PATH = "output_videos/juggle_counter.mp4"
BALL_CLASS_ID = 0  # update if different
FOOT_CLASS_ID = 1
DIST_THRESHOLD = 100
COOLDOWN_FRAMES = 10

model = YOLO('models/best.pt')
cap = cv2.VideoCapture('input_videos/youtube1.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) # how why how???

out = cv2.VideoWriter('output_videos/juggle_counter.mp4', fourcc, fps, (width, height))

juggle_count = 0
was_above = True
is_below = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, device=0, verbose=False, persist=True)

    foot_top_y, ball_bottom_y = None, None

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        box_id = box.id
        

        # Draw bounding boxes
        label = "Foot" if cls_id == FOOT_CLASS_ID else "Ball" if cls_id == BALL_CLASS_ID else str(cls_id)
        color = (0, 255, 0) if cls_id == FOOT_CLASS_ID else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} ID: {int(box_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # print('Label', label, '| box_id', box_id)

        if cls_id == FOOT_CLASS_ID:
            foot_top_y = y1  # top of foot bounding box
        elif cls_id == BALL_CLASS_ID:
            ball_bottom_y = y2  # bottom of ball bounding box

    if foot_top_y is not None and ball_bottom_y is not None:

        if not is_below and ball_bottom_y > foot_top_y:
            is_below = True
        elif is_below and ball_bottom_y < foot_top_y:
            is_below = False
            juggle_count += 1

    # Overlay juggle count on frame
    cv2.putText(frame, f'Juggles: {juggle_count}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # Write frame to output
    out.write(frame)

cap.release()
out.release()
print(f"final juggle count: {juggle_count}")
print(f"video saved at: {OUTPUT_PATH}")