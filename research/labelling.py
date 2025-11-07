#!/usr/bin/env python3
"""
Interactive labelling tool for LED line detection target lines.

This tool provides a simple OpenCV-based GUI to draw and edit target lines
for each test image. Target lines are saved as JSON files that can be used
by the evaluation pipeline.

Usage:
    python labelling.py --images images --targets targets

Controls:
    - Click and drag to draw a new line
    - Press 'd' to delete the last line
    - Press 's' to save and move to next image
    - Press 'q' to quit without saving
    - Press 'n' to skip to next image without saving
    - Press 'p' to go back to previous image
"""

import cv2
import numpy as np
import json
import os
import argparse
from typing import List, Optional, Tuple, Dict


class TargetLineLabeller:
    """Interactive tool for labelling target lines in images."""

    def __init__(self, images_folder: str, targets_folder: str):
        self.images_folder = images_folder
        self.targets_folder = targets_folder

        # Load all images
        image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        self.image_files = sorted(
            [
                f
                for f in os.listdir(images_folder)
                if f.lower().endswith(image_extensions)
            ]
        )

        if not self.image_files:
            raise ValueError(f"No images found in {images_folder}")

        # State
        self.current_index = 0
        self.current_image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self.lines: List[Dict] = []
        self.drawing = False
        self.start_point: Optional[Tuple[int, int]] = None
        self.temp_line: Optional[Tuple[int, int, int, int]] = None

        # UI settings
        self.window_name = "LED Line Labeller"
        self.line_color = (0, 255, 0)  # Green
        self.line_thickness = 2
        self.temp_line_color = (255, 255, 0)  # Cyan for line being drawn

        # Create targets folder if it doesn't exist
        os.makedirs(targets_folder, exist_ok=True)

    def load_image(self):
        """Load the current image and its existing targets if any."""
        image_file = self.image_files[self.current_index]
        image_path = os.path.join(self.images_folder, image_file)

        self.current_image = cv2.imread(image_path)
        if self.current_image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Load existing targets if available
        basename = os.path.splitext(image_file)[0]
        targets_file = os.path.join(self.targets_folder, f"{basename}.json")

        self.lines = []
        if os.path.exists(targets_file):
            try:
                with open(targets_file, "r") as f:
                    data = json.load(f)
                    target_lines = data.get("target_lines", [])
                    for target in target_lines:
                        line_info = target["approximate_line"]
                        self.lines.append(
                            {
                                "id": target["id"],
                                "name": target.get("name", f"Line {target['id']}"),
                                "x1": line_info["x1"],
                                "y1": line_info["y1"],
                                "x2": line_info["x2"],
                                "y2": line_info["y2"],
                                "tolerance": target.get("tolerance", 20),
                                "description": target.get("description", ""),
                            }
                        )
                print(f"Loaded {len(self.lines)} existing lines from {targets_file}")
            except Exception as e:
                print(f"Warning: Could not load targets from {targets_file}: {e}")

    def draw_lines(self):
        """Draw all lines on the display image."""
        assert self.current_image is not None, "Current image must be loaded"
        self.display_image = self.current_image.copy()

        # Draw all saved lines
        for i, line in enumerate(self.lines):
            pt1 = (line["x1"], line["y1"])
            pt2 = (line["x2"], line["y2"])
            cv2.line(self.display_image, pt1, pt2, self.line_color, self.line_thickness)

            # Draw line ID
            mid_x = (line["x1"] + line["x2"]) // 2
            mid_y = (line["y1"] + line["y2"]) // 2
            cv2.putText(
                self.display_image,
                f"L{line['id']}",
                (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Draw temporary line being drawn
        if self.temp_line:
            x1, y1, x2, y2 = self.temp_line
            cv2.line(
                self.display_image,
                (x1, y1),
                (x2, y2),
                self.temp_line_color,
                self.line_thickness,
            )

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing lines."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing a new line
            self.drawing = True
            self.start_point = (x, y)
            self.temp_line = None

        elif event == cv2.EVENT_MOUSEMOVE:
            # Update temporary line while drawing
            if self.drawing and self.start_point and self.display_image is not None:
                self.temp_line = (self.start_point[0], self.start_point[1], x, y)
                self.draw_lines()
                cv2.imshow(self.window_name, self.display_image)

        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing the line
            if self.drawing and self.start_point and self.display_image is not None:
                # Add the line to the list
                next_id = max([line["id"] for line in self.lines], default=0) + 1
                self.lines.append(
                    {
                        "id": next_id,
                        "name": f"Line {next_id}",
                        "x1": self.start_point[0],
                        "y1": self.start_point[1],
                        "x2": x,
                        "y2": y,
                        "tolerance": 20,
                        "description": "",
                    }
                )

                self.drawing = False
                self.start_point = None
                self.temp_line = None
                self.draw_lines()
                cv2.imshow(self.window_name, self.display_image)

    def save_targets(self):
        """Save the current target lines to JSON."""
        assert self.current_image is not None, "Current image must be loaded"
        image_file = self.image_files[self.current_index]
        basename = os.path.splitext(image_file)[0]
        targets_file = os.path.join(self.targets_folder, f"{basename}.json")

        h, w = self.current_image.shape[:2]

        data = {
            "description": f"Target lines for {image_file}",
            "image": image_file,
            "image_width": w,
            "image_height": h,
            "target_lines": [
                {
                    "id": line["id"],
                    "name": line["name"],
                    "approximate_line": {
                        "x1": line["x1"],
                        "y1": line["y1"],
                        "x2": line["x2"],
                        "y2": line["y2"],
                    },
                    "tolerance": line["tolerance"],
                    "description": line["description"],
                }
                for line in self.lines
            ],
            "scoring_notes": None,
        }

        with open(targets_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(self.lines)} lines to {targets_file}")

    def get_status_text(self):
        """Get status text for display."""
        image_file = self.image_files[self.current_index]
        return f"Image {self.current_index + 1}/{len(self.image_files)}: {image_file} | Lines: {len(self.lines)} | [s]ave [d]elete [n]ext [p]rev [q]uit"

    def run(self):
        """Run the interactive labelling tool."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n" + "=" * 80)
        print("LED Line Labeller")
        print("=" * 80)
        print("\nControls:")
        print("  - Click and drag to draw a new line")
        print("  - Press 'd' to delete the last line")
        print("  - Press 's' to save and move to next image")
        print("  - Press 'n' to skip to next image without saving")
        print("  - Press 'p' to go back to previous image")
        print("  - Press 'q' to quit")
        print()

        self.load_image()
        self.draw_lines()

        while True:
            # Add status text to image
            assert self.display_image is not None, "Display image must be initialized"
            display = self.display_image.copy()
            status = self.get_status_text()
            cv2.putText(
                display,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                # Quit
                print("\nQuitting...")
                break

            elif key == ord("s"):
                # Save and move to next image
                self.save_targets()
                if self.current_index < len(self.image_files) - 1:
                    self.current_index += 1
                    self.load_image()
                    self.draw_lines()
                else:
                    print("\nReached last image!")

            elif key == ord("n"):
                # Next image without saving
                if self.current_index < len(self.image_files) - 1:
                    self.current_index += 1
                    self.load_image()
                    self.draw_lines()
                else:
                    print("\nReached last image!")

            elif key == ord("p"):
                # Previous image
                if self.current_index > 0:
                    self.current_index -= 1
                    self.load_image()
                    self.draw_lines()
                else:
                    print("\nAlready at first image!")

            elif key == ord("d"):
                # Delete last line
                if self.lines:
                    deleted = self.lines.pop()
                    print(f"Deleted line {deleted['id']}")
                    self.draw_lines()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive tool for labelling target lines in LED detection images."
    )
    parser.add_argument(
        "--images",
        type=str,
        default="images",
        help="Path to folder containing images to label (default: images)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="targets",
        help="Path to folder to save target JSON files (default: targets)",
    )

    args = parser.parse_args()

    # Check if images folder exists
    if not os.path.exists(args.images):
        print(f"Error: Images folder not found: {args.images}")
        return

    # Run the labeller
    labeller = TargetLineLabeller(args.images, args.targets)
    labeller.run()


if __name__ == "__main__":
    main()
