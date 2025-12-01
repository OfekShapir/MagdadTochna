import os
import cv2
import time
from collections import defaultdict
from discover_cards_frames import discover_cards
from april_tags_frames import detect_apriltags
import math
# Setup folders
os.makedirs("results/photos", exist_ok=True)
os.makedirs("results/texts", exist_ok=True)
os.makedirs("results/marked", exist_ok=True)

# RUN ID SYSTEM
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

# our digital camera calibration data
K = [
    [1.39561099e+03, 0.00000000e+00, 8.85690305e+02],
    [0.00000000e+00, 1.38830766e+03, 5.04754597e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
]
D = [-0.07011441, 0.24724181, 0.00124205, -0.00364551, -0.27059026]

fx = K[0][0]
fy = K[1][1]
cx = K[0][2]
cy = K[1][2]

camera_params = [fx, fy, cx, cy]

# ----------------------------------------------------
# every frames goes to both functions
# ----------------------------------------------------


def distance(first_location, second_location):
    deltax=first_location[0] - second_location[0]
    deltay=first_location[1] - second_location[1]
    return math.sqrt(deltax**2 + deltay**2)

def find_closest(april_poses, card_poses, found_cards, num_of_cards, tag_id=1):
    if not found_cards:
        return [], []
    cards=[]
    dis=[]
    found_cards_copy=found_cards.copy()
    for i in range(num_of_cards):
        j = 0
        for card in found_cards_copy:
            if j==0:
                j+=1
                closest_card =card
                closest_dis = distance(april_poses[tag_id], card_poses[closest_card])
            temp_dis = distance(april_poses[tag_id], card_poses[card])
            if(closest_dis > temp_dis):
                closest_dis= temp_dis
                closest_card = card
        cards.append(closest_card)
        dis.append(closest_dis)
        found_cards_copy.remove(closest_card)
    return cards,dis

def game(name):
    print("lets play " + name)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise Exception("Could not open camera.")

    frame_id = 0
    last_time = time.time()
    CAPTURE_INTERVAL = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 1) detect AprilTags (and draw them)
        frame_with_tags, april_poses, found_tags = detect_apriltags(frame,camera_params)
        now = time.time()
        save_now = False
        if now - last_time >= CAPTURE_INTERVAL:
            last_time = now
            frame_id += 1
            save_now = True
            # Save original frame for history if you want
            photo_path = f"results/photos/frame_{RUN_ID}_{frame_id:04d}.jpg"
            cv2.imwrite(photo_path, frame)
        # run YOLO card detection
        annotated_frame, card_poses, found_cards = discover_cards(
            frame_with_tags,
            frame_id,
            RUN_ID,
            save_outputs=save_now
        )

        # Show the LIVE annotated view
        cv2.imshow("Magic: Cards + AprilTags", annotated_frame)
        num_of_cards= int(input("How many cards do we have?"))
        cards,distances = find_closest(april_poses, card_poses, found_cards,num_of_cards)
        print(cards)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


game("bj")
cv2.destroyAllWindows()
