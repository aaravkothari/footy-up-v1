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

# Change Output Path and Video Capture Input Path

OUTPUT_PATH = "output_videos/juggle_counter_capture.mp4"
BALL_CLASS_ID = 0  # update if different
FOOT_CLASS_ID = 1
SAVE_OUTPUT = True

model = YOLO('models/best.pt')
# cap = cv2.VideoCapture('input_videos/youtube1.mp4')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

juggle_count = 0
was_above = True # old algorithm
is_below = False
foot_top_y = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, device=0, verbose=False)

    foot_top_y, ball_bottom_y = None, None

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        

        # Draw bounding boxes
        label = "Foot" if cls_id == FOOT_CLASS_ID else "Ball" if cls_id == BALL_CLASS_ID else str(cls_id)
        
        color = (0, 255, 0) if cls_id == FOOT_CLASS_ID else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # print('Label', label, '| box_id', box_id)

        if cls_id == FOOT_CLASS_ID:
            if foot_top_y == None:
                foot_top_y = y1  # top of foot bounding box
                print('Found Foot')
            else:
                if foot_top_y > y1:
                    print('Lower Foot:', foot_top_y)
                    foot_top_y = y1
                    print('Higher Foot:', foot_top_y)

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
    cv2.imshow("Juggle Counter", frame)

    if SAVE_OUTPUT:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

if SAVE_OUTPUT:
    out.release()
    print(f"video saved at: {OUTPUT_PATH}")

cv2.destroyAllWindows()
print(f"final juggle count: {juggle_count}")