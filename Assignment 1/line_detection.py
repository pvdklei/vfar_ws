# WORK IN PROGRESS

import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_in_frame(frame, bin_t=250, min_length=50, max_gap=15):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, bin_t, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.Canny(image, 0, 255)
    lines = cv2.HoughLinesP(image, 1, np.pi/180, 30, minLineLength=min_length, maxLineGap=max_gap)
    return lines

def display_frame(frame, lines):
    pass

def main():
    pass

if __name__ == "__main__":
    main()