import cv2
import os
import time
from ultralytics import YOLO
from collections import defaultdict
from drawimages import DrawImages
cards_model = YOLO("yolov8s_playing_cards.pt")
import math

def found_card(data):
    stats=data[0]["confidence"]**2 + data[1]["confidence"]**2
    if (stats>1):
        return True
    return False

def center_of_one(data):
    top_left = (data[0]["edges"][0],data[0]["edges"][1])
    top_right = (data[0]["edges"][2], data[0]["edges"][1])
    bottom_left = (data[0]["edges"][0],data[0]["edges"][3])
    bottom_right = (data[0]["edges"][2], data[0]["edges"][3])




def discover_cards(frame, output_id, RUN_ID, save_outputs=False):
    """
    Runs YOLO on a single frame (numpy array), finds cards,
    computes per-card positions, and optionally saves results.

    card_poses[label] = [x, y]  (image coordinates)
    found_cards = list(card_poses.keys())
    """

    # work on a copy so we can draw
    img = frame.copy()

    # -------------------------------
    # NEW: store drawing until after detection
    # -------------------------------
    drawing_ops = []   # list of functions that draw AFTER all cards are found

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

        # ---------------------------------------------------------------
        # INSTEAD OF DRAWING NOW, SAVE A DRAWING TASK FOR LATER
        # Draw card center + label (red)
        # ---------------------------------------------------------------
        drawing_ops.append(
            DrawImages(cx, cy, label, (0, 0, 255), box=(x1, y1, x2, y2)).draw_card
        )
    # Group by label
    label_groups = defaultdict(list)
    for card in cards:
        label_groups[card["label"]].append(card)

    pair_centers = []   # middle points for duplicated cards
    card_poses = {}     # final per-card position

    # Middle points for duplicated cards
    for label, data in label_groups.items():
        if len(data) > 2:
            data.sort(key=lambda x: x["confidence"], reverse=True)
        if len(data) >= 2:
            if(found_card(data)):
                c1, c2 = data[0]["center"], data[1]["center"]
                mid_x = (c1[0] + c2[0]) / 2
                mid_y = (c1[1] + c2[1]) / 2

                pair_centers.append({"Card": label, "middle": (mid_x, mid_y)})
                card_poses[label] = [mid_x, mid_y]
                drawing_ops.append(
                    DrawImages(mid_x, mid_y, label, (0, 255, 0)).draw_card
                )
        elif len(data)==1:
            # TODO FIND A WAY TO FIND THE CENTER
            #center_for_one(data)
            ...
    # For labels that appear only once (or more than twice), use the single center
    found_cards = list(card_poses.keys())

    # ---------------------------------------
    # DRAW EVERYTHING *NOW*, AFTER ALL FOUND
    # ---------------------------------------
    for draw in drawing_ops:
        draw(img)

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



def pixel_to_camera(u, v, Z, fx, fy, cx, cy):
    """
    Convert a pixel (u,v) at depth Z into camera coordinates (X,Y,Z) in meters.
    Assumes pinhole camera model and intrinsics fx, fy, cx, cy.
    """
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return [X, Y, Z]


