import os
import cv2
import time
import sys
from collections import defaultdict
import math
from discover_cards_frames import discover_cards,pixel_to_camera
from april_tags_frames import detect_apriltags
def distance_checker():
    """
    Detect all cards + all AprilTags in a single frame,
    compute 3D distances between each card and each tag.

    Returns:
        {
            card_label: {
                tag_id: distance_in_meters,
                tag_id: distance_in_meters,
                ...
            },
            ...
        },
        april_poses_3d,   # {tag_id: [X,Y,Z]}
        card_poses_3d     # {card_label: [X,Y,Z]}
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise Exception("Could not open camera.")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise Exception("Could not capture frame.")

    cap.release()

    # 1) Detect AprilTags
    frame_for_tags = frame.copy()
    frame_with_tags, april_poses, found_tags = detect_apriltags(
        frame_for_tags,
        camera_params
    )

    # Convert AprilTags to 3D (already returned as 3D, but ensure dict is clean)
    april_poses_3d = {
        tag_id: pose for tag_id, pose in april_poses.items()
    }

    # 2) Detect cards (YOLO)
    annotated_frame, card_poses, found_cards = discover_cards(
        frame_with_tags,
        frame_id=0,
        RUN_ID=RUN_ID,
        save_outputs=False
    )

    # 3) Convert card pixels → real 3D camera coordinates
    card_poses_3d = {}
    if tag_id_for_depth in april_poses_3d:
        Z_ref = april_poses_3d[tag_id_for_depth][2]  # z of tag in meters

        for label, (u, v) in card_poses.items():
            X, Y, Z = pixel_to_camera(u, v, Z_ref, fx, fy, cx, cy)
            card_poses_3d[label] = [X, Y, Z]

    # 4) Compute distances between every card and every tag
    distances = {}

    for card_label, card_xyz in card_poses_3d.items():

        distances[card_label] = {}

        for tag_id, tag_xyz in april_poses_3d.items():
            dx = card_xyz[0] - tag_xyz[0]
            dy = card_xyz[1] - tag_xyz[1]
            dz = card_xyz[2] - tag_xyz[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            distances[card_label][tag_id] = dist

    return distances, april_poses_3d, card_poses_3d
