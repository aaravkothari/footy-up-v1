from ultralytics import YOLO

model = YOLO("yolo11x.pt")

results = model.predict(source="input_videos/youtube1.mp4", device=0, save=True)
print(results[0])
print("==================")
for box in results[0].boxes:
    print(box)

    