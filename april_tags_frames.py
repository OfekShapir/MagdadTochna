from pupil_apriltags import Detector
import cv2

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
TAG_SIZE=0.08
def detect_apriltags(frame,camera_params):
    """
    Runs AprilTag detection on a single frame, draws the tags,
    and returns april_poses and list of found tag IDs.

    april_poses[tag_id] = [x, y, z]  (in meters, camera coordinate frame)
    """

    april_poses = {}

    # rgb to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # scan gray photos for squares
    results = detector.detect(
        gray,
        estimate_tag_pose=True,  # Don't just find the tag in 2D; do the complex linear algebra to figure out its 3D position.
        camera_params=camera_params,  # use what we forced
        tag_size=TAG_SIZE,
    )

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
        t = r.pose_t  # matrix 3x1
        if t is not None:
            x, y, z = t.flatten()  # x left right, y up down, z depth
            april_poses[r.tag_id] = [x, y, z]

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

    found_tags = list(april_poses.keys())
    return frame, april_poses, found_tags
