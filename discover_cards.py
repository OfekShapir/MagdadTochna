from ultralytics import YOLO
import cv2
from collections import defaultdict

model = YOLO("Playing-Cards-Detection/yolov8s_playing_cards.pt")

results = model.predict(
    "poker.png",
    iou=0.6,
    imgsz=1280,
    max_det=200,
    half=False,
    augment=True
)

res = results[0]

# Load the original image
img = cv2.imread("poker.png")

# Store detected cards
cards = []

for box in res.boxes:
    cls_id = int(box.cls)
    conf = float(box.conf)
    label = res.names[cls_id]

    # bounding box
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    # center of bounding box of the number telling the card
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    cards.append({
        "label": label,
        "confidence": conf,
        "center": (cx, cy)
    })

# ----------------------------
# GROUP CARDS BY LABEL
# ----------------------------

label_groups = defaultdict(list)
for card in cards:
    label_groups[card["label"]].append(card)

# ----------------------------
# COMPUTE MIDDLE POINT FOR PAIRS
# ----------------------------

pair_centers = []  # 2D array for output

for label, group in label_groups.items():
    if len(group) == 2:  # exactly two cards with same name
        c1 = group[0]["center"]
        c2 = group[1]["center"]

        # midpoint:
        mid_x = (c1[0] + c2[0]) / 2
        mid_y = (c1[1] + c2[1]) / 2

        # store in array
        pair_centers.append({
            "Card": label,
            "middle": (mid_x, mid_y)
        })

        # draw the middle point on the image
        cv2.circle(img, (int(mid_x), int(mid_y)), 10, (0, 255, 0), -1)
        cv2.putText(img, label, (int(mid_x)+10, int(mid_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

# Save final image
cv2.imwrite("output_cards_centers.jpg", img)
print("Cards Boundries:")
for card in cards:
    print(card)
print("Cards that has 2 boundries' centers:")
for card in pair_centers:
    print(card)