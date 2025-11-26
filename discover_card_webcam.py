import cv2
import os
import time
from ultralytics import YOLO
from collections import defaultdict

# ----------------------------------------------------
# Setup folders
# ----------------------------------------------------
os.makedirs("results/photos", exist_ok=True)
os.makedirs("results/texts", exist_ok=True)
os.makedirs("results/marked", exist_ok=True)

# Unique run ID using timestamp
import os

# ----------------------------------------------------
# RUN ID SYSTEM
# ----------------------------------------------------
RUN_FILE = "run_id.txt"

# Create file if missing
if not os.path.exists(RUN_FILE):
    with open(RUN_FILE, "w") as f:
        f.write("0")

# Increment run ID
with open(RUN_FILE, "r") as f:
    RUN_ID = int(f.read().strip()) + 1

with open(RUN_FILE, "w") as f:
    f.write(str(RUN_ID))

print(f"Starting RUN #{RUN_ID}")


# Load YOLO model
model = YOLO("Playing-Cards-Detection/yolov8s_playing_cards.pt")

# ----------------------------------------------------
#  DISCOVER CARDS
# ----------------------------------------------------
def discover_cards(image_or_frame, output_id, live=False):
    """
    If live=True → image_or_frame is a frame (numpy array)
    If live=False → image_or_frame is a file path
    """

    if live:
        img = image_or_frame.copy()
    else:
        img = cv2.imread(image_or_frame)

    # YOLO detection
    results = model.predict(
        img,
        iou=0.5,
        imgsz=1280,
        max_det=20,
        half=False,
        augment=False,###########change to True if it makes bugs. should teohreticly make the image proccesing better
        verbose=False
    )
    res = results[0]
    cards = []

    # Extract card data
    for box in res.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = res.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        cards.append({
            "label": label,
            "confidence": conf,
            "edges": (x1, y1, x2, y2),
            "center": (cx, cy)
        })

        # Draw card center
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.putText(img, label, (int(cx) + 10, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Group by label
    label_groups = defaultdict(list)
    for card in cards:
        label_groups[card["label"]].append(card)

    pair_centers = []

    # Middle points for duplicated cards
    for label, group in label_groups.items():
        if len(group) == 2:
            c1, c2 = group[0]["center"], group[1]["center"]
            mid_x = (c1[0] + c2[0]) / 2
            mid_y = (c1[1] + c2[1]) / 2

            pair_centers.append({"Card": label, "middle": (mid_x, mid_y)})

            cv2.circle(img, (int(mid_x), int(mid_y)), 5, (0, 255, 0), -1)
            cv2.putText(img, label, (int(mid_x) + 15, int(mid_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save outputs (only for saved frames, not live frames)
    if not live:
        mark_path = f"results/marked/frame_{RUN_ID}_{output_id:04d}_marked.jpg"
        text_path = f"results/texts/frame_{RUN_ID}_{output_id:04d}.txt"

        cv2.imwrite(mark_path, img)

        with open(text_path, "w") as f:
            f.write(f"Run {RUN_ID}, Frame {output_id}\n")
            f.write("===========================\n\n")
            f.write("Detected cards:\n\n")
            for c in cards:
                f.write(f"Card: {c['label']}\n")
                f.write(f"Edges: {c['edges']}\n")
                f.write(f"Center: {c['center']}\n\n")

            f.write("\nPairs (2 cards):\n")
            for p in pair_centers:
                f.write(f"{p['Card']}: middle = {p['middle']}\n")

            # set of labels that appear in pairs
            paired_labels = {p["Card"] for p in pair_centers}

            f.write("\n1-Card (no pair center):\n\n")

            for c in cards:
                if c["label"] not in paired_labels:
                    f.write(f"Card: {c['label']}\n")
                    f.write(f"Edges: {c['edges']}\n")
                    f.write(f"Center: {c['center']}\n\n")

    return img  # Return annotated image (used for live display)



# ----------------------------------------------------
# WEBCAM LOOP — 2 frames per second
# ----------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise Exception("Could not open camera.")

frame_id = 0
last_time = time.time()
CAPTURE_INTERVAL = 0.2  # 2 fps

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Live processed frame (drawn markings)
    live_annotated = discover_cards(frame, frame_id, live=True)

    now = time.time()
    if now - last_time >= CAPTURE_INTERVAL:
        last_time = now
        frame_id += 1

        # Save original frame
        photo_path = f"results/photos/frame_{RUN_ID}_{frame_id:04d}.jpg"
        cv2.imwrite(photo_path, frame)

        # Process and save file outputs
        discover_cards(photo_path, frame_id, live=False)

    # Show the LIVE annotated view
    cv2.imshow("Card Detector Live", live_annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
