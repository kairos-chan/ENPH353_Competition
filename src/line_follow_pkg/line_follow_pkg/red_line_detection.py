## Red line detection: finds the red line and returns the y of its lowest point
import cv2
import numpy as np
 
 
def detect_red_line_y(frame, min_area=150):
    # frame: RGB image from the front camera.
    # Returns the y of the lowest point of the red line, or None if no red line is visible.
 
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
 
    # red wraps around the hue axis, so use two ranges
    mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
 
    # noise removal: remove small red blobs that are not the line
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
 
    # take the largest area of red and ignore small red noise
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None
 
    line = max(contours, key=cv2.contourArea)
    ys = line[:, 0, 1]          # y of every point on the blob's outline
    return int(ys.max())        # lowest point = highest y
 
 
def detect_red_in_right_third(frame, min_pixels=500):
    # frame: RGB image from the front camera.
    # Returns True if there is red in the right third of the image, else False.
 
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
 
    # red wraps around the hue axis, so use two ranges
    mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
 
    # noise removal: remove small red blobs
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
 
    height, width = mask.shape[:2]
    bottom_right = mask[height // 2:, (2 * width) // 3:]
    return int(np.count_nonzero(bottom_right)) >= min_pixels
