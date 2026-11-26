"""Pedestrian detection: finds the jean-blue pants and tells when the pedestrian
has left the frame to the left or to the right."""
import cv2
import numpy as np
 
JEAN_LOW = (100, 60, 30)
JEAN_HIGH = (130, 255, 200)
 
 
def blue_mask(frame):
    """frame: RGB image from the front camera. Returns the cleaned jean-blue mask (mono8)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, JEAN_LOW, JEAN_HIGH)
 
    # noise removal, then close small gaps so the legs form one blob
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask
 
 
def find_pedestrian(mask, min_area=300):
    # Returns (x_min, x_max, cx, y_min, y_max) of the largest blue blob, or None if no blob is found.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None
 
    blob = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(blob)
    return x, x + w, x + w / 2.0, y, y + h
 
 
def detect_pedestrian(frame, min_area=300):
    return find_pedestrian(blue_mask(frame), min_area)
 
 
def draw_debug(frame, mask, det, label=''):
    ## Returns an RGB debug image: blue pixels tinted, green box around the pedestrian.
    debug = frame.copy()
    debug[mask > 0] = (0.5 * debug[mask > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
 
    if det is not None:
        x_min, x_max, cx, y_min, y_max = det
        cv2.rectangle(debug, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.circle(debug, (int(cx), int((y_min + y_max) / 2)), 5, (255, 255, 0), -1)
        cv2.putText(debug, 'pedestrian', (int(x_min), max(15, int(y_min) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
 
    if label:
        cv2.putText(debug, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return debug
 
 
class PedestrianTracker:
    """Call update(det, width) on every front-camera frame while waiting.
 
    det is the result of find_pedestrian(). Returns None while the pedestrian is still
    around (or hasn't been seen yet), and 'left' or 'right' once the pedestrian has
    walked out of that side of the frame.
    """
 
    def __init__(self, edge_margin=8, missing_frames=5):
        self.edge_margin = edge_margin        # px from the image border that counts as at the edge
        self.missing_frames = missing_frames  # consecutive empty frames before we say "gone"
        self.reset()
 
    def reset(self):
        self.seen = False
        self.missing = 0
        self.last_box = None   # (x_min, x_max, cx, ...) from the last frame it was visible
 
    def update(self, det, width):
        if det is not None:
            self.seen = True
            self.missing = 0
            self.last_box = det
            return None
 
        if not self.seen:
            return None
 
        self.missing += 1
        if self.missing < self.missing_frames:
            return None
 
        # pedestrian has been gone for a while: detects which side they left from
        x_min, x_max = self.last_box[0], self.last_box[1]
        side = None
        if x_min <= self.edge_margin:
            side = 'left'
        elif x_max >= width - self.edge_margin:
            side = 'right'
 
        # if it vanished mid-frame (noise/occlusion), don't count it as an exit:
        # forget it and keep waiting for a proper detection.
        self.reset()
        return side