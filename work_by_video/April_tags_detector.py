import cv2
from pupil_apriltags import Detector

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
TAG_SIZE = 0.08  # 8 cm tag

# the april tag detector
detector = Detector(
    # our type of april tag, contains 36 bits, atleast 11 bits diff between ids.
    families="tag36h11",
    # use 4 cpu cores to make detection faster. most cpus have 4-8 cores so its ok for computer need check on raspberry
    nthreads=4,
    # shrink img by half. pros faster cons miss small tags
    #quad_decimate=2.0,
    quad_decimate=1.0,

    #after a tag is found it goes back to full size to make edges
    refine_edges=True,
    decode_sharpening=0.25,
)

#open the camera
cap = cv2.VideoCapture(0)
# force cv to run on our camera proportion 1920 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# take one image to check if its crashing
ret, frame = cap.read()
if not ret:
    raise Exception("Could not read from camera")

poses = {}   # will hold {tag_id: (x,y,z)}
# start an infinity loop to process frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # rgb to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # scan gray photos for squares
    results = detector.detect(
        gray,
        estimate_tag_pose=True, # Don't just find the tag in 2D; do the complex linear algebra to figure out its 3D position.
        camera_params=camera_params, # use what we forced
        tag_size=TAG_SIZE,
    )
    poses = {}
    for r in results:
        # draw corners
        for (px, py) in r.corners:
            cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)

        # tag ID
        cv2.putText(
            frame,
            f"ID {r.tag_id}",
            (int(r.center[0]) - 20, int(r.center[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        # pose (safe)
        t = r.pose_t # matrix 3x1
        if t is not None:
            x, y, z = t.flatten() # x left right, y up down, z depth
            poses[r.tag_id] = (x, y, z)
            cv2.putText(
                frame,
                f"z={z:.3f} m",
                (int(r.center[0]) - 20, int(r.center[1]) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"x={x:.3f} m",
                (int(r.center[0]) - 20, int(r.center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"y={y:.3f} m",
                (int(r.center[0]) - 20, int(r.center[1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Pose failed",
                (int(r.center[0]) - 20, int(r.center[1]) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    cv2.imshow("AprilTag Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
