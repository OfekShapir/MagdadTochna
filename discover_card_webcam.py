import cv2
from ultralytics import YOLO

# Load your trained card detector
model = YOLO("yolov8s_playing_cards.pt")

# Open default webcam (index 0)
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise Exception("Could not open camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(
        frame,
        conf=0.10,
        iou=0.7,
        imgsz=1280,
        verbose=False
    )

    # Draw the boxes on the frame
    annotated_frame = results[0].plot()  # YOLO draws boxes for us

    # Show on screen
    cv2.imshow("Card Detector", annotated_frame)

    # Close on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
