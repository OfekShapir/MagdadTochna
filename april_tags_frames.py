from pupil_apriltags import Detector
import cv2
from drawimages import DrawImages

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
    drawing_ops = []

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
        # ------------------------------------------
        # Extract pose
        # ------------------------------------------
        t = r.pose_t
        if t is not None:
            x, y, z = t.flatten()
            april_poses[r.tag_id] = [x, y, z]
        else:
            april_poses[r.tag_id] = None  # still returned but no pose

        # ------------------------------------------
        # Bounding box for AprilTag
        # r.corners is 4 points:
        # p0, p1, p2, p3  -> get min/max to form box
        # ------------------------------------------
        xs = [int(p[0]) for p in r.corners]
        ys = [int(p[1]) for p in r.corners]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        # ------------------------------------------
        # Create drawing object
        # ------------------------------------------
        tag_label = f"ID {r.tag_id}"

        draw_tag = DrawImages(
            x=r.center[0],
            y=r.center[1],
            label=tag_label,
            color=(255, 0, 0),  # blue-ish for tags
            box=(x1, y1, x2, y2),
        )

        # Schedule tag drawing (rectangle + text + center)
        drawing_ops.append(draw_tag.draw_tag)

        # ------------------------------------------
        # draw corners as green dots
        # ------------------------------------------
        for (px, py) in r.corners:
            highlight = DrawImages(px, py, "", (0, 255, 0))
            drawing_ops.append(highlight.draw_card)  # small green dot


    for draw in drawing_ops:
        draw(frame)

    found_tags = list(april_poses.keys())
    return frame, april_poses, found_tags
