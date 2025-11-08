# WORK IN PROGRESS

import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Assume you run this in a terminal and are in the Assignment 1 folder of the repo
# (for automatic default path arguments)
PATH: str = os.getcwd()

def detect_in_frame(frame, bin_t=250, edge="canny", votes=70, min_length=50, max_gap=20):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, bin_t, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    image = cv2.dilate(image, kernel, iterations=2)
    if edge == "canny":
        image = cv2.Canny(image, 0, 255)
    elif edge == "laplacian":
        image = cv2.Laplacian(image, cv2.CV_64F)
        image = cv2.convertScaleAbs(image)
    else:
        pass
    lines = cv2.HoughLinesP(image, 1, np.pi/180, votes, minLineLength=min_length, maxLineGap=max_gap)
    return lines

def save_frame(filename, frame, lines, dest):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imwrite(f"{dest}/{filename.split(".")[0]}_lines.{filename.split(".")[1]}", frame)

def main(args):
    # Get list image path(s)
    paths: list = []
    if len(args.src.split(".")) < 2:
        # This is a folder with a series of frames
        paths = list(os.walk(args.src))[0][-1]
    else:
        # This is a single image
        paths.append(args.src)

    # Open images & run line detection for each
    for path in paths:
        image = cv2.imread(f"{args.src}/{path}")
        lines = detect_in_frame(image, args.bin_t, args.edge, args.votes, args.min_length, args.max_gap)
        save_frame(path, image, lines, args.dest)

if __name__ == "__main__":
    print(PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", help="Path to input image or folder", default=PATH + "/images")
    parser.add_argument("-d", "--dest", help="Path to output folder", default=PATH + "/output")
    parser.add_argument("-t", "--bin_t", help="Threshold for binarizing the image", default=251)
    parser.add_argument("-e", "--edge", help="Edge detection method (e.g.: Canny or Laplacian)", default="canny")
    parser.add_argument("-v", "--votes", help="HoughLines parameter of min. nr. votes", default=70)
    parser.add_argument("--min_length", help="HoughLines parameter of min. line length", default=50)
    parser.add_argument("--max_gap", help="HoughLines parameter of max. line gap", default=20)
    args = parser.parse_args()
    main(args)