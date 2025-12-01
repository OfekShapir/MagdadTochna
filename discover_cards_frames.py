import cv2
import os
import time
from ultralytics import YOLO
from collections import defaultdict

cards_model = YOLO("yolov8s_playing_cards.pt")

def discover_cards(frame, output_id,RUN_ID,save_outputs=False):
    """
    Runs YOLO on a single frame (numpy array), finds cards,
    computes per-card positions, and optionally saves results.

    card_poses[label] = [x, y]  (image coordinates)
    found_cards = list(card_poses.keys())
    """

    # work on a copy so we can draw
    img = frame.copy()

    # YOLO detection
    results = cards_model.predict(
        img,
        iou=0.5,
        imgsz=1280,
        max_det=20,
        half=False,
        augment=False,  #change to True if it makes bugs. should teohreticly make the image proccesing better
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

    pair_centers = []   # middle points for duplicated cards
    card_poses = {}     # final per-card position

    # Middle points for duplicated cards
    for label, group in label_groups.items():
        if len(group) == 2:
            c1, c2 = group[0]["center"], group[1]["center"]
            mid_x = (c1[0] + c2[0]) / 2
            mid_y = (c1[1] + c2[1]) / 2

            pair_centers.append({"Card": label, "middle": (mid_x, mid_y)})

            # This is the "logical" card position when we have 2 copies
            card_poses[label] = [mid_x, mid_y]

            cv2.circle(img, (int(mid_x), int(mid_y)), 5, (0, 255, 0), -1)
            cv2.putText(img, label, (int(mid_x) + 15, int(mid_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # For labels that appear only once (or more than twice), use the single center
    for card in cards:
        label = card["label"]
        if label not in card_poses:
            cx, cy = card["center"]
            card_poses[label] = [cx, cy]

    found_cards = list(card_poses.keys())

    # Save outputs (only when we ask for saved frames, not every frame)
    if save_outputs:
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

    return img, card_poses, found_cards

