# WORK IN PROGRESS

import os
import cv2
import numpy as np
import argparse
from cv_bridge import CvBridge
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

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

def save_frame(frame_id, frame, lines, dest):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    out_path = os.path.join(dest, f"frame_{frame_id}.png")
    cv2.imwrite(out_path, frame)

def process_rosbag(bag_path, dest, **kwargs):
    bridge = CvBridge()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    frame_id = 0

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == "/image_raw":
            # deserialize the message
            msg = deserialize_message(data, Image)
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # apply detector
            lines = detect_in_frame(frame, **kwargs)
            save_frame(frame_id, frame, lines, dest)

            frame_id += 1
            if frame_id % 50 == 0:
                print(f"Processed {frame_id} frames...")

    print(f"✅ Finished: {frame_id} frames processed and saved to {dest}")

def main(args):
    os.makedirs(args.dest, exist_ok=True)

    # If src is a ROS bag
    if args.src.endswith(".db3"):
        process_rosbag(
            args.src,
            args.dest,
            bin_t=args.bin_t,
            edge=args.edge,
            votes=args.votes,
            min_length=args.min_length,
            max_gap=args.max_gap
        )
    else:
        # else on folder with images
        paths = []
        if len(args.src.split(".")) < 2:
            paths = list(os.walk(args.src))[0][-1]
        else:
            paths.append(args.src)

        frame_id = 0
        for path in paths:
            image = cv2.imread(f"{args.src}/{path}")
            lines = detect_in_frame(image, args.bin_t, args.edge, args.votes,
                                    args.min_length, args.max_gap)
            save_frame(frame_id, image, lines, args.dest)
            frame_id += 1

if __name__ == "__main__":
    print(PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", help="Path to input (folder, image, or .db3 rosbag)", default=PATH + "/images")
    parser.add_argument("-d", "--dest", help="Path to output folder", default=PATH + "/output")
    parser.add_argument("-t", "--bin_t", help="Threshold for binarizing the image", default=251)
    parser.add_argument("-e", "--edge", help="Edge detection method (e.g.: Canny or Laplacian)", default="canny")
    parser.add_argument("-v", "--votes", help="HoughLines parameter of min. nr. votes", default=70)
    parser.add_argument("--min_length", help="HoughLines parameter of min. line length", default=50)
    parser.add_argument("--max_gap", help="HoughLines parameter of max. line gap", default=20)
    args = parser.parse_args()
    main(args)