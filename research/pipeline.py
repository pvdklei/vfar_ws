import os
import cv2
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import product
from tqdm import tqdm
import json
import shutil


class LineDetectionPipeline:
    """
    A comprehensive line detection pipeline with grid search capabilities.

    The pipeline detects bright LED lines through:
    1. Brightness thresholding and filtering
    2. Edge detection (Canny, Sobel)
    3. Line detection (Hough Transform, LSD)
    """

    def __init__(self, image_path: str, output_folder: str, targets_path: Optional[str] = None):
        self.image_path = image_path
        self.output_folder = output_folder
        self.original_color = cv2.imread(image_path)

        # Pre-compute color space conversions
        self.gray = cv2.cvtColor(self.original_color, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.original_color, cv2.COLOR_BGR2HSV)

        # Load target lines if provided
        self.targets = None
        if targets_path and os.path.exists(targets_path):
            with open(targets_path, 'r') as f:
                self.targets = json.load(f)

    def apply_brightness_threshold(self, colorspace: str, threshold: int,
                                   use_adaptive: bool = False, use_clahe: bool = False,
                                   use_background_norm: bool = False, roi_top_ratio: float = 0.0) -> np.ndarray:
        """Apply brightness threshold with optional preprocessing enhancements and ROI masking."""
        if colorspace == 'gray':
            channel = self.gray.copy()
        elif colorspace == 'hsv':
            channel = self.hsv[:, :, 2].copy()
        else:
            raise ValueError(f"Unknown colorspace: {colorspace}")

        # Step 1: Background normalization (remove slow-varying illumination)
        if use_background_norm:
            channel_float = channel.astype(np.float32)
            blur = cv2.GaussianBlur(channel_float, (31, 31), 0)
            # Subtract background and normalize
            normalized = channel_float - blur
            channel = cv2.normalize(normalized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

        # Step 2: Apply CLAHE for better contrast
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            channel = clahe.apply(channel)

        # Step 3: Thresholding (adaptive or global)
        if use_adaptive:
            mask = cv2.adaptiveThreshold(
                channel, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,  # block size
                -5  # C constant
            )
        else:
            _, mask = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY)

        # Step 4: Apply ROI mask (focus on top portion of image for ceiling LEDs)
        if roi_top_ratio > 0:
            roi_mask = np.zeros_like(mask)
            height = mask.shape[0]
            roi_height = int(height * roi_top_ratio)
            roi_mask[:roi_height, :] = 255
            mask = cv2.bitwise_and(mask, roi_mask)

        return mask

    def apply_morphology(self, mask: np.ndarray, kernel_size: int, iterations: int,
                        use_closing: bool = False) -> np.ndarray:
        """Apply morphological operations with proper border handling."""
        if kernel_size > 0 and iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            # Use closing (dilation + erosion) to connect gaps without excessive thickening
            if use_closing:
                result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations,
                                         borderType=cv2.BORDER_CONSTANT)
            else:
                result = cv2.dilate(mask, kernel, iterations=iterations, borderType=cv2.BORDER_CONSTANT)
            # Ensure output has same shape as input
            assert result.shape == mask.shape, f"Shape mismatch: {result.shape} vs {mask.shape}"
            return result
        return mask

    def create_overlay(self, mask: np.ndarray, use_overlay: bool) -> np.ndarray:
        """Create overlay of mask on original image or return mask."""
        if use_overlay:
            return cv2.bitwise_and(self.original_color, self.original_color, mask=mask)
        return mask

    def detect_edges(self, image: np.ndarray, method: str, **params) -> np.ndarray:
        """Detect edges using specified method with optional Gaussian blur."""
        # Optional: Apply Gaussian blur before edge detection to reduce noise
        blur_kernel = params.get('blur_kernel', 0)
        if blur_kernel > 0:
            # Ensure kernel is odd
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

        if method == 'canny':
            # Ensure grayscale for Canny (works on overlay or mask)
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.dtype != np.uint8:
                normalized = np.zeros_like(image, dtype=np.uint8)
                image = cv2.normalize(image, normalized, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            threshold1 = params.get('canny_low', 50)
            threshold2 = params.get('canny_high', 150)
            edges = cv2.Canny(image, threshold1, threshold2)
            # Ensure edges have same spatial dimensions as original
            expected_shape = (self.original_color.shape[0], self.original_color.shape[1])
            assert edges.shape == expected_shape, f"Edge shape mismatch: {edges.shape} vs {expected_shape}"
            return edges
        elif method == 'sobel':
            # Sobel edge detection
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            # Guard against div-by-zero
            max_val = magnitude.max()
            if max_val > 0:
                magnitude_normalized = magnitude / max_val * 255
            else:
                magnitude_normalized = np.zeros_like(magnitude, dtype=np.uint8)
            magnitude_uint8 = magnitude_normalized.astype(np.uint8)
            # Apply threshold to get binary edges
            _, edges = cv2.threshold(magnitude_uint8, params.get('sobel_threshold', 50), 255, cv2.THRESH_BINARY)
            return edges
        else:
            raise ValueError(f"Unknown edge detection method: {method}")

    def merge_collinear_lines(self, lines: Optional[np.ndarray],
                             angle_threshold: float = 5.0,
                             distance_threshold: float = 30.0) -> Optional[np.ndarray]:
        """Merge collinear line segments into longer lines."""
        if lines is None or len(lines) <= 1:
            return lines

        merged = []
        used = set()

        for i, line1 in enumerate(lines):
            if i in used:
                continue

            x1, y1, x2, y2 = line1[0]
            angle1 = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Collect all lines similar to this one
            similar_lines = [(x1, y1, x2, y2)]
            used.add(i)

            for j, line2 in enumerate(lines):
                if j in used:
                    continue

                x3, y3, x4, y4 = line2[0]
                angle2 = np.degrees(np.arctan2(y4 - y3, x4 - x3))

                # Check angle similarity
                angle_diff = abs(angle1 - angle2)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                if angle_diff > angle_threshold:
                    continue

                # Check if lines are close (point to line distance)
                dist1 = self.point_to_line_distance(x3, y3, x1, y1, x2, y2)
                dist2 = self.point_to_line_distance(x4, y4, x1, y1, x2, y2)

                if max(dist1, dist2) < distance_threshold:
                    similar_lines.append((x3, y3, x4, y4))
                    used.add(j)

            # Merge all similar lines by fitting through all endpoints
            if len(similar_lines) > 1:
                all_points = []
                for line in similar_lines:
                    all_points.append([line[0], line[1]])
                    all_points.append([line[2], line[3]])
                all_points = np.array(all_points)

                # Fit line through all points using PCA
                mean = np.mean(all_points, axis=0)
                centered = all_points - mean
                _, _, vt = np.linalg.svd(centered)
                direction = vt[0]

                # Project all points onto the line and find extremes
                projections = np.dot(centered, direction)
                min_proj = np.min(projections)
                max_proj = np.max(projections)

                # Calculate endpoints
                start = mean + min_proj * direction
                end = mean + max_proj * direction

                merged.append([[int(start[0]), int(start[1]), int(end[0]), int(end[1])]])
            else:
                merged.append(line1)

        return np.array(merged) if merged else None

    def detect_lines(self, edges: np.ndarray, method: str, **params) -> Optional[np.ndarray]:
        """Detect lines using specified method."""
        if method == 'hough':
            threshold = params.get('hough_threshold', 100)
            min_line_length = params.get('min_line_length', 100)
            max_line_gap = params.get('max_line_gap', 10)
            return cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=threshold,
                minLineLength=min_line_length,
                maxLineGap=max_line_gap
            )
        elif method == 'lsd':
            # Line Segment Detector
            lsd = cv2.createLineSegmentDetector(0)
            lines, _, _, _ = lsd.detect(edges)
            if lines is not None:
                # Convert to HoughLinesP format for consistency
                lines = lines.reshape(-1, 1, 4).astype(int)
            return lines
        else:
            raise ValueError(f"Unknown line detection method: {method}")

    def draw_lines(self, image: np.ndarray, lines: Optional[np.ndarray],
                   matched_targets: Optional[set] = None, qualities: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
        """
        Draw lines on image with quality-based color coding.

        Color scheme:
        - Green: Matched target lines
        - Red: Low quality unmatched lines (q <= 0.3)
        - Yellow: Medium quality unmatched lines (0.3 < q < 0.6)
        - Orange: High quality unmatched lines (q >= 0.6)
        """
        result = image.copy()
        count = 0
        if lines is not None:
            for idx, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]

                if matched_targets and idx in matched_targets:
                    # Green for matched targets
                    color = (0, 255, 0)
                else:
                    # Color-code by quality for unmatched lines
                    if qualities is not None:
                        q = qualities[idx]
                        if q <= 0.3:
                            color = (0, 0, 255)      # Red: bad quality
                        elif q < 0.6:
                            color = (0, 255, 255)    # Yellow: medium quality
                        else:
                            color = (0, 165, 255)    # Orange: high quality but unmatched
                    else:
                        color = (0, 0, 255)  # Default red for unmatched

                cv2.line(result, (x1, y1), (x2, y2), color, 2)
                count += 1
        return result, count

    def point_to_line_distance(self, px: float, py: float, x1: float, y1: float,
                               x2: float, y2: float) -> float:
        """Calculate perpendicular distance from point to line segment."""
        line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
        if line_len_sq == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)

        # Parameter t represents position along line segment (0 to 1)
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))

        # Find projection point on line
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)

        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def line_matches_target(self, detected_line: Tuple[int, int, int, int],
                           target_line: Dict, tolerance: float) -> bool:
        """Check if a detected line matches a target line within tolerance, including angle."""
        dx1, dy1, dx2, dy2 = detected_line
        tx1 = target_line['approximate_line']['x1']
        ty1 = target_line['approximate_line']['y1']
        tx2 = target_line['approximate_line']['x2']
        ty2 = target_line['approximate_line']['y2']

        # Calculate angles (in degrees)
        detected_angle = np.degrees(np.arctan2(dy2 - dy1, dx2 - dx1))
        target_angle = np.degrees(np.arctan2(ty2 - ty1, tx2 - tx1))

        # Normalize angles to [-180, 180]
        detected_angle = ((detected_angle + 180) % 360) - 180
        target_angle = ((target_angle + 180) % 360) - 180

        # Calculate angle difference (accounting for wraparound)
        angle_diff = abs(detected_angle - target_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # STRICT: Reject if angle difference > 15 degrees
        if angle_diff > 15:
            return False

        # Check if endpoints of detected line are close to target line
        dist1 = self.point_to_line_distance(dx1, dy1, tx1, ty1, tx2, ty2)
        dist2 = self.point_to_line_distance(dx2, dy2, tx1, ty1, tx2, ty2)

        # Also check if target line endpoints are close to detected line
        dist3 = self.point_to_line_distance(tx1, ty1, dx1, dy1, dx2, dy2)
        dist4 = self.point_to_line_distance(tx2, ty2, dx1, dy1, dx2, dy2)

        # Line matches if endpoints are close AND angle is similar
        avg_dist = (dist1 + dist2 + dist3 + dist4) / 4
        return avg_dist < tolerance

    def _sample_line_brightness(self, V: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                                width: int = 3, num_samples: int = 100) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Sample brightness along a line and in perpendicular strips for background comparison.

        Returns:
            (line_values, bg_values): Arrays of brightness values, or (None, None) if line is invalid
        """
        xs = np.linspace(x1, x2, num_samples)
        ys = np.linspace(y1, y2, num_samples)
        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)

        if length == 0:
            return None, None

        # Unit perpendicular vector (normal)
        nx = -dy / length
        ny = dx / length

        line_vals = []
        bg_vals = []

        H, W = V.shape
        for x, y in zip(xs, ys):
            x_c = int(round(x))
            y_c = int(round(y))
            if 0 <= x_c < W and 0 <= y_c < H:
                line_vals.append(V[y_c, x_c])

                # Sample both sides perpendicular to line
                for k in range(1, width + 1):
                    xp1 = int(round(x + nx * k))
                    yp1 = int(round(y + ny * k))
                    xp2 = int(round(x - nx * k))
                    yp2 = int(round(y - ny * k))
                    if 0 <= xp1 < W and 0 <= yp1 < H:
                        bg_vals.append(V[yp1, xp1])
                    if 0 <= xp2 < W and 0 <= yp2 < H:
                        bg_vals.append(V[yp2, xp2])

        if not line_vals:
            return None, None
        return np.array(line_vals), np.array(bg_vals) if bg_vals else None

    def compute_line_quality(self, line: np.ndarray, bright_thr: int = 180) -> float:
        """
        Compute a quality score [0,1] for a detected line based on brightness profile.

        High quality lines (LED strips) should be:
        - Very bright along their length
        - Consistent brightness (low std)
        - Brighter than surrounding background

        Args:
            line: Line coordinates [x1, y1, x2, y2]
            bright_thr: Threshold for "bright" pixels

        Returns:
            Quality score in [0, 1] where 1 = perfect LED candidate
        """
        x1, y1, x2, y2 = line[0]
        V = self.hsv[:, :, 2]  # Brightness channel

        line_vals, bg_vals = self._sample_line_brightness(V, x1, y1, x2, y2)

        if line_vals is None:
            return 0.0

        mean_line = float(line_vals.mean())
        std_line = float(line_vals.std())
        bright_ratio = float((line_vals > bright_thr).mean())

        if bg_vals is not None and len(bg_vals) > 0:
            mean_bg = float(bg_vals.mean())
            contrast = mean_line - mean_bg
        else:
            mean_bg = 0.0
            contrast = mean_line

        # Normalize each component to [0, 1] with soft thresholds
        q_bright = np.clip((mean_line - 150) / 80.0, 0.0, 1.0)       # 150-230 range
        q_ratio = np.clip((bright_ratio - 0.6) / 0.4, 0.0, 1.0)      # 60-100% bright pixels
        q_contrast = np.clip((contrast - 20) / 80.0, 0.0, 1.0)       # 20-100 contrast
        q_stable = 1.0 - np.clip((std_line - 10) / 40.0, 0.0, 1.0)   # Low std => stable

        # Weighted combination
        q = 0.35 * q_bright + 0.25 * q_ratio + 0.25 * q_contrast + 0.15 * q_stable
        return float(q)

    def compute_lines_quality(self, lines: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Compute quality scores for all lines."""
        if lines is None:
            return None
        qualities = []
        for line in lines:
            q = self.compute_line_quality(line)
            qualities.append(q)
        return np.array(qualities)

    def evaluate_lines(self, lines: Optional[np.ndarray], qualities: Optional[np.ndarray] = None,
                      q_good: float = 0.6, q_bad: float = 0.3) -> Dict:
        """
        Evaluate detected lines with quality-based scoring.

        Args:
            lines: Detected lines
            qualities: Quality scores for each line [0,1], or None to compute them
            q_good: Threshold for "good" LED candidates (>=)
            q_bad: Threshold for "bad" lines (<=)

        Quality-based classification:
            - TP: Matched target with quality >= q_good
            - FP_bad: Unmatched line with quality <= q_bad (strong penalty)
            - FP_soft: Unmatched line with q_bad < quality < q_good (mild penalty)
            - FN: Unmatched target
        """
        if lines is None or len(lines) == 0:
            return {
                'count': 0,
                'matched_targets': 0,
                'matched_target_ids': [],
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'score': 0,
                'matched_line_indices': set(),
                'qualities': []
            }

        # Compute qualities if not provided
        if qualities is None:
            qualities = self.compute_lines_quality(lines)

        # If no targets provided, fall back to simple counting
        if not self.targets:
            return {
                'count': len(lines),
                'matched_targets': 0,
                'matched_target_ids': [],
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'score': len(lines) * 10,
                'matched_line_indices': set(),
                'qualities': qualities.tolist() if qualities is not None else []
            }

        target_lines = self.targets.get('target_lines', [])
        matched_targets = set()
        matched_target_ids = []
        matched_line_indices = set()

        # For each target, find if any detected line matches it
        # Only count as TP if line quality >= q_good
        for target in target_lines:
            target_id = target['id']
            tolerance = target.get('tolerance', 20)

            for idx, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                if self.line_matches_target((x1, y1, x2, y2), target, tolerance):
                    # Only accept match if quality is good
                    if qualities is not None and qualities[idx] >= q_good:
                        matched_targets.add(target_id)
                        matched_target_ids.append(target_id)
                        matched_line_indices.add(idx)
                        break  # Move to next target once matched

        # Classify false positives by quality
        fp_bad = 0
        fp_soft = 0
        for idx in range(len(lines)):
            if idx not in matched_line_indices:
                q = qualities[idx] if qualities is not None else 0.5
                if q <= q_bad:
                    fp_bad += 1
                elif q < q_good:
                    fp_soft += 1
                # Note: high-quality unmatched lines (q >= q_good) don't count as bad FPs

        num_detected = len(lines)
        num_targets = len(target_lines)
        tp = len(matched_targets)  # True positives
        fn = num_targets - tp      # False negatives

        # Calculate precision and recall with quality weighting
        # Precision: TP / (TP + FP_bad + FP_soft)
        total_fp = fp_bad + fp_soft
        precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F-beta score: beta < 1 gives more weight to precision
        beta = 0.7
        if precision + recall > 0:
            f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        else:
            f_beta = 0

        # Strong exponential penalty for bad false positives
        alpha = 0.5  # Penalty strength
        bad_penalty = np.exp(-alpha * fp_bad)

        # Score: Precision-weighted F-score with exponential bad-FP penalty
        score = f_beta * bad_penalty * 1000

        return {
            'count': num_detected,
            'matched_targets': tp,
            'matched_target_ids': matched_target_ids,
            'precision': precision,
            'recall': recall,
            'f1_score': f_beta,  # Using F-beta score (precision-weighted)
            'fp_bad': fp_bad,
            'fp_soft': fp_soft,
            'score': score,
            'matched_line_indices': matched_line_indices,
            'qualities': qualities.tolist() if qualities is not None else []
        }

    def process_configuration(self, config: Dict) -> Tuple[Dict, List[Tuple[str, np.ndarray]]]:
        """Process a single configuration and return results."""
        steps = []

        # Step 1: Brightness threshold with preprocessing
        mask = self.apply_brightness_threshold(
            config['colorspace'],
            config['brightness_threshold'],
            use_adaptive=config.get('use_adaptive', False),
            use_clahe=config.get('use_clahe', False),
            use_background_norm=config.get('use_background_norm', False),
            roi_top_ratio=config.get('roi_top_ratio', 0.0)
        )
        steps.append(('1_brightness_mask', mask))

        # Step 2: Morphology
        mask = self.apply_morphology(
            mask,
            config['morph_kernel_size'],
            config['morph_iterations'],
            use_closing=config.get('use_closing', False)
        )
        steps.append(('2_morphology', mask))

        # Step 3: Overlay (or use mask)
        processed = self.create_overlay(mask, config['use_overlay'])
        steps.append(('3_overlay_or_mask', processed))

        # Step 4: Edge detection
        edges = self.detect_edges(
            processed,
            config['edge_method'],
            canny_low=config.get('canny_low', 50),
            canny_high=config.get('canny_high', 150),
            blur_kernel=config.get('blur_kernel', 0),
            sobel_threshold=config.get('sobel_threshold', 50)
        )
        steps.append(('4_edges', edges))

        # Step 5: Line detection
        lines = self.detect_lines(
            edges,
            config['line_method'],
            hough_threshold=config.get('hough_threshold', 100),
            min_line_length=config.get('min_line_length', 100),
            max_line_gap=config.get('max_line_gap', 10)
        )

        # Step 5.5: Merge collinear lines if enabled
        if config.get('merge_lines', False):
            lines = self.merge_collinear_lines(lines)

        # Step 5.6: Compute line quality scores
        qualities = self.compute_lines_quality(lines) if lines is not None else None

        # Evaluate with quality scores
        metrics = self.evaluate_lines(lines, qualities=qualities)

        # Step 6: Draw lines (color-coded by quality and match status)
        matched_indices = metrics.get('matched_line_indices', set())
        result, _ = self.draw_lines(self.original_color, lines, matched_indices, qualities=qualities)
        steps.append(('5_final_lines', result))

        metrics['config'] = config

        return metrics, steps

    def save_results(self, config_id: int, config: Dict, metrics: Dict, steps: List[Tuple[str, np.ndarray]]):
        """Save all images for a configuration."""
        # Create folder for this configuration
        config_name = self.generate_config_name(config)
        folder_path = os.path.join(self.output_folder, f"{config_id:03d}_{config_name}")
        os.makedirs(folder_path, exist_ok=True)

        # Save all intermediate steps
        for step_name, image in steps:
            cv2.imwrite(os.path.join(folder_path, f"{step_name}.png"), image)

        # Save configuration and metrics
        info = {
            'config': config,
            'metrics': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else
                          list(v) if isinstance(v, set) else v)
                       for k, v in metrics.items() if k not in ['config', 'matched_line_indices']}
        }
        with open(os.path.join(folder_path, 'info.json'), 'w') as f:
            json.dump(info, f, indent=2)

        return folder_path

    def generate_config_name(self, config: Dict) -> str:
        """Generate a human-readable name encoding all important config parameters."""
        # Build preprocessing flags
        flags = []
        if config.get('use_background_norm'): flags.append("bg")
        if config.get('use_clahe'):           flags.append("cl")
        if config.get('use_adaptive'):        flags.append("ad")
        if config.get('blur_kernel', 0) > 0:  flags.append(f"b{config['blur_kernel']}")
        if config.get('merge_lines'):         flags.append("mg")
        flags_str = "-".join(flags) if flags else "plain"

        parts = [
            config['colorspace'],
            f"thr{config['brightness_threshold']}",
            f"k{config['morph_kernel_size']}i{config['morph_iterations']}",
            "ovr" if config['use_overlay'] else "msk",
            config['edge_method'],
            config['line_method'],
        ]

        # Add edge/line detection params
        if config.get('canny_low'):
            parts.append(f"cl{config['canny_low']}")
        if config.get('canny_high'):
            parts.append(f"ch{config['canny_high']}")
        if config.get('hough_threshold'):
            parts.append(f"h{config['hough_threshold']}")
        if config.get('min_line_length'):
            parts.append(f"ml{config['min_line_length']}")

        # Add ROI if used
        if config.get('roi_top_ratio', 0) > 0:
            parts.append(f"roi{int(config['roi_top_ratio']*100)}")

        # Add preprocessing flags at the end
        parts.append(flags_str)

        return "_".join(str(p) for p in parts if p)

    def draw_target_lines(self, save_path: str):
        """Draw target lines on original image and save visualization."""
        if not self.targets:
            return

        result = self.original_color.copy()
        target_lines = self.targets.get('target_lines', [])

        for target in target_lines:
            line_info = target['approximate_line']
            x1, y1 = line_info['x1'], line_info['y1']
            x2, y2 = line_info['x2'], line_info['y2']
            target_id = target['id']

            # Draw target line in blue
            cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Add label with ID
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(result, f"T{target_id}", (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imwrite(save_path, result)
        print(f"Target lines visualization saved to: {save_path}")


def create_grid_search_configs() -> List[Dict]:
    """
    Create grid search configurations.

    Optimized for detecting multiple LED light strips on ceiling.
    """
    configs = []

    # Define search space - comprehensive preprocessing pipeline
    colorspaces = ['hsv']  # HSV-V works best for brightness
    brightness_thresholds = [160, 180]  # Test around sweet spot

    # NEW: Advanced preprocessing options
    use_background_norms = [False, True]  # Test illumination normalization
    use_clahes = [False, True]  # Test contrast enhancement
    use_adaptives = [False, True]  # Test adaptive thresholding

    # ROI masking - focus on ceiling region only
    roi_top_ratios = [0.0, 0.5, 0.6]  # 0=full image, 0.5=top 50%, 0.6=top 60%

    morph_kernel_sizes = [7, 9]  # Larger kernels work better
    morph_iterations = [2]  # Balance
    use_closings = [True]  # Closing works better than dilation
    use_overlays = [True]  # Overlay preserves edges
    edge_methods = ['canny']  # Canny is best
    line_methods = ['hough']  # Focus on Hough for now
    merge_lines_options = [False, True]  # Test line merging

    # Hough parameters - stricter to reduce false positives
    hough_thresholds = [30, 40]  # Higher threshold
    min_line_lengths = [60]  # Longer segments only
    max_line_gaps = [50]  # Moderate gap tolerance

    # Canny parameters
    canny_lows = [15, 20]  # Lower for detection
    canny_highs = [100]  # Keep moderate

    # Gaussian blur before edge detection
    blur_kernels = [0, 3]  # No blur vs light blur

    # Generate all combinations
    for (colorspace, brightness_threshold, use_bg_norm, use_clahe, use_adaptive, roi_top_ratio,
         kernel_size, iterations, use_closing, use_overlay, edge_method, line_method, merge_lines) in product(
        colorspaces, brightness_thresholds, use_background_norms, use_clahes, use_adaptives, roi_top_ratios,
        morph_kernel_sizes, morph_iterations, use_closings, use_overlays, edge_methods, line_methods,
        merge_lines_options
    ):
        # Base configuration
        config = {
            'colorspace': colorspace,
            'brightness_threshold': brightness_threshold,
            'use_background_norm': use_bg_norm,
            'use_clahe': use_clahe,
            'use_adaptive': use_adaptive,
            'roi_top_ratio': roi_top_ratio,
            'morph_kernel_size': kernel_size,
            'morph_iterations': iterations,
            'use_closing': use_closing,
            'use_overlay': use_overlay,
            'edge_method': edge_method,
            'line_method': line_method,
            'merge_lines': merge_lines,
        }

        # Add method-specific parameters
        if edge_method == 'canny':
            for canny_low, canny_high, blur_kernel in product(canny_lows, canny_highs, blur_kernels):
                config_copy = config.copy()
                config_copy['canny_low'] = canny_low
                config_copy['canny_high'] = canny_high
                config_copy['blur_kernel'] = blur_kernel

                if line_method == 'hough':
                    for hough_threshold, min_length, max_gap in product(
                        hough_thresholds, min_line_lengths, max_line_gaps
                    ):
                        config_final = config_copy.copy()
                        config_final['hough_threshold'] = hough_threshold
                        config_final['min_line_length'] = min_length
                        config_final['max_line_gap'] = max_gap
                        configs.append(config_final)
                else:
                    configs.append(config_copy)

    print(f"Generated {len(configs)} configurations for grid search")
    return configs


def main():
    """
    Main pipeline with grid search for optimal line detection parameters.

    Detects LED light lines by testing various combinations of:
    - Color spaces (Gray, HSV)
    - Brightness thresholds
    - Morphological operations
    - Edge detection methods (Canny, Sobel)
    - Line detection methods (Hough Transform, LSD)
    """

    parser = argparse.ArgumentParser(
        description="Line detection pipeline with grid search capabilities."
    )
    parser.add_argument(
        "input_image",
        nargs='?',
        default="field.png",
        type=str,
        help="Path to the input image (default: field.png)"
    )
    parser.add_argument(
        "output_folder",
        nargs='?',
        default=os.path.join("output", "linedetect"),
        type=str,
        help="Path to save results (default: output/linedetect)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top configurations to save (default: 10)"
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="targets.json",
        help="Path to targets JSON file (default: targets.json)"
    )
    parser.add_argument(
        "--preview-targets",
        action="store_true",
        help="Only generate target_lines_reference.png and exit (for quick editing)"
    )
    args = parser.parse_args()

    # Clean output folder (delete everything inside it for a fresh start)
    if os.path.exists(args.output_folder):
        print(f"Cleaning output folder: {args.output_folder}")
        shutil.rmtree(args.output_folder)

    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # Initialize pipeline
    print(f"Loading image: {args.input_image}")
    targets_path = args.targets if os.path.exists(args.targets) else None
    if targets_path:
        print(f"Loading targets from: {targets_path}")
    else:
        print("No targets file found, using generic scoring")
    pipeline = LineDetectionPipeline(args.input_image, args.output_folder, targets_path)

    # Draw target lines visualization if targets are provided
    if targets_path:
        target_viz_path = os.path.join(args.output_folder, "target_lines_reference.png")
        pipeline.draw_target_lines(target_viz_path)

    # If preview mode, exit after drawing targets
    if args.preview_targets:
        print("\nPreview mode: Target visualization complete. Exiting.")
        return

    # Generate configurations
    print("\n" + "="*80)
    print("GENERATING GRID SEARCH CONFIGURATIONS")
    print("="*80)
    configs = create_grid_search_configs()

    # Process all configurations
    print("\n" + "="*80)
    print("PROCESSING CONFIGURATIONS")
    print("="*80)
    results = []

    with tqdm(total=len(configs), desc="Overall Progress", position=0, leave=True,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for i, config in enumerate(configs):
            # Create a sub-progress bar for this configuration
            with tqdm(total=6, desc=f"Config {i+1}", position=1, leave=False,
                     bar_format='{desc}: {bar}| {n_fmt}/{total_fmt}') as subpbar:
                subpbar.set_description(f"Config {i+1}: Brightness")
                subpbar.update(1)

                subpbar.set_description(f"Config {i+1}: Morphology")
                subpbar.update(1)

                subpbar.set_description(f"Config {i+1}: Overlay")
                subpbar.update(1)

                subpbar.set_description(f"Config {i+1}: Edges")
                subpbar.update(1)

                subpbar.set_description(f"Config {i+1}: Lines")
                metrics, steps = pipeline.process_configuration(config)
                subpbar.update(1)

                subpbar.set_description(f"Config {i+1}: Evaluate")
                subpbar.update(1)

            results.append({
                'config_id': i,
                'config': config,
                'metrics': metrics,
                'steps': steps
            })

            pbar.update(1)
            # Update the main progress bar with current best score
            if results:
                best_score = max(r['metrics']['score'] for r in results)
                pbar.set_postfix({'best_score': f'{best_score:.1f}'})

    # Sort by score (descending)
    print("\n\nSorting results by score...")
    results.sort(key=lambda x: x['metrics']['score'], reverse=True)

    # Save top-k results
    print("\n" + "="*80)
    print(f"SAVING TOP {args.top_k} CONFIGURATIONS")
    print("="*80)
    top_results = []

    with tqdm(total=args.top_k, desc="Saving Results", position=0,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        for rank, result in enumerate(results[:args.top_k]):
            folder = pipeline.save_results(
                rank + 1,
                result['config'],
                result['metrics'],
                result['steps']
            )

            metrics_to_save = {
                'rank': rank + 1,
                'folder': folder,
                'score': result['metrics']['score'],
                'line_count': result['metrics']['count'],
                'config': result['config']
            }

            # Add target-based metrics if available
            if 'matched_targets' in result['metrics']:
                metrics_to_save.update({
                    'matched_targets': result['metrics']['matched_targets'],
                    'matched_target_ids': result['metrics']['matched_target_ids'],
                    'precision': result['metrics']['precision'],
                    'recall': result['metrics']['recall'],
                    'f1_score': result['metrics']['f1_score']
                })

            top_results.append(metrics_to_save)

            # Update progress bar
            postfix = {'rank': rank + 1, 'score': f"{result['metrics']['score']:.1f}"}
            if 'matched_targets' in result['metrics']:
                postfix['matched'] = f"{result['metrics']['matched_targets']}/{result['metrics']['count']}"
            pbar.set_postfix(postfix)
            pbar.update(1)

    # Save summary
    summary_path = os.path.join(args.output_folder, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(top_results, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("TOP CONFIGURATIONS")
    print("="*80)
    for res in top_results[:5]:
        print(f"\nRank {res['rank']}: Score={res['score']:.1f}")
        print(f"  Lines Detected: {res['line_count']}")

        # Show target-based metrics if available
        if 'matched_targets' in res:
            total_targets = len(pipeline.targets['target_lines']) if pipeline.targets else 0
            print(f"  Target Matches: {res['matched_targets']}/{total_targets} targets")
            print(f"  Matched IDs: {res['matched_target_ids']}")
            print(f"  Precision: {res['precision']:.1%}, Recall: {res['recall']:.1%}, F1: {res['f1_score']:.3f}")

        print(f"  Config: {pipeline.generate_config_name(res['config'])}")
        print(f"  Folder: {res['folder']}")

    print(f"\nFull summary saved to: {summary_path}")
    print(f"Total configurations tested: {len(configs)}")
    print(f"Top {args.top_k} results saved to: {args.output_folder}")


if __name__ == "__main__":
    main()
