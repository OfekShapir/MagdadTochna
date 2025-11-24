from ultralytics import YOLO

model = YOLO("yolov8s_playing_cards.pt")

results = model.predict(
    "poker.jpg",
    iou=0.6,
    imgsz=1280,
    max_det=200,     # ensures NMS doesn't remove cards
    half=False,      # avoid precision loss on CPU
    augment=True     # improves robustness
)


cards = []
res = results[0]

for box in res.boxes:
    cls_id = int(box.cls)
    conf = float(box.conf)
    label = res.names[cls_id]
    cards.append({"label": label, "confidence": conf})

print(cards)
