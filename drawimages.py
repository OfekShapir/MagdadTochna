import cv2

class DrawImages:
    def __init__(self, x, y, label, color, box=None):
        """
        x, y     = center of the object (cx, cy)
        label    = text to draw
        color    = (B,G,R)
        box      = optional (x1, y1, x2, y2) bounding box
                   used for tags OR for placing label inside card box
        """
        self.x = int(x)
        self.y = int(y)
        self.label = str(label)
        self.color = color      # (B,G,R)
        self.box = box          # optional bounding-box

    # ------------------------------------------------------
    # HELPER: Ensures the label stays inside the bounding box
    # ------------------------------------------------------
    def smart_label_position(self):
        if self.box is None:
            return (self.x + 10, self.y)  # fallback

        x1, y1, x2, y2 = self.box
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        # Try placing at top-left inside bounding box
        tx = int(x1 + 5)
        ty = int(y1 + 20)

        # If the box is too small, fallback to center
        if width < 40 or height < 40:
            return (self.x + 10, self.y)

        return (tx, ty)

    # ------------------------------------------------------
    # CARDS: simple circle + label
    # ------------------------------------------------------
    def draw_card(self, image):
        cv2.circle(image, (self.x, self.y), 6, self.color, -1)

        text_pos = self.smart_label_position()
        cv2.putText(
            image,
            self.label,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.color,
            2
        )

    # ------------------------------------------------------
    # APRIL TAG: draw a box + center circle + label inside box
    # ------------------------------------------------------
    def draw_tag(self, image):
        if self.box:
            x1, y1, x2, y2 = map(int, self.box)
            cv2.rectangle(image, (x1, y1), (x2, y2), self.color, 2)

        cv2.circle(image, (self.x, self.y), 5, self.color, -1)

        text_pos = self.smart_label_position()
        cv2.putText(
            image,
            self.label,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.color,
            2
        )
