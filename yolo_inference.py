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

# === CONFIGURATION ===
MODEL_PATH = "models/best.pt"
VIDEO_PATH = "input_videos/youtube1.mp4"
OUTPUT_PATH = "output_videos/juggle_counter.mp4"
BALL_CLASS_ID = 0  # update if different
FOOT_CLASS_ID = 1
DIST_THRESHOLD = 100
COOLDOWN_FRAMES = 10

# === INIT ===
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# Video settings
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Make output folder if needed
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

juggle_count = 0
cooldown = 0

def get_center(box):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    return (x1 + x2) / 2, (y1 + y2) / 2

# === PROCESS VIDEO FRAME-BY-FRAME ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, device=0, verbose=False)[0]

    foot_center, ball_center = None, None
    for box in results.boxes:
        cls_id = int(box.cls[0])
        center = get_center(box)
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        # Draw bounding boxes
        label = "Foot" if cls_id == FOOT_CLASS_ID else "Ball" if cls_id == BALL_CLASS_ID else str(cls_id)
        color = (0, 255, 0) if cls_id == FOOT_CLASS_ID else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if cls_id == FOOT_CLASS_ID:
            foot_center = center
        elif cls_id == BALL_CLASS_ID:
            ball_center = center

    # Calculate and update juggle count
    if foot_center and ball_center:
        distance = np.linalg.norm(np.array(foot_center) - np.array(ball_center))

        if distance < DIST_THRESHOLD and cooldown == 0:
            juggle_count += 1
            cooldown = COOLDOWN_FRAMES
        elif cooldown > 0:
            cooldown -= 1

    # Overlay juggle count on frame
    cv2.putText(frame, f'Juggles: {juggle_count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Write frame to output
    out.write(frame)

# === CLEANUP ===
cap.release()
out.release()
print(f"✅ Done! Final juggle count: {juggle_count}")
print(f"📁 Video saved at: {OUTPUT_PATH}")


    