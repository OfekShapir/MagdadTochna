from ultralytics import YOLO

# load the pretrained model
model = YOLO("Playing-Cards-Detection/yolov8s_playing_cards.pt")

# run prediction on your table image
results = model("poker.jpg")[0]

cards = []
for box in results.boxes:
    cls_id = int(box.cls.item())
    conf = float(box.conf.item())
    label = results.names[cls_id]  # e.g. "AH", "10C", "KS"
    cards.append({"label": label, "confidence": conf})

print(cards)
