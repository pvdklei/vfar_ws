import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, TypedDict, Literal, Union
import json


def select_channel(gray: np.ndarray, hsv: np.ndarray, colorspace: Literal['gray', 'hsv']) -> np.ndarray:
    """
    Select appropriate image channel based on colorspace configuration.
    
    Args:
        gray: Grayscale image
        hsv: HSV image 
        colorspace: Either 'gray' or 'hsv'
        
    Returns:
        Selected channel as numpy array
        
    Raises:
        ValueError: If colorspace is not 'gray' or 'hsv'
    """
    if colorspace == 'gray':
        return gray.copy()
    elif colorspace == 'hsv':
        return hsv[:, :, 2].copy()  # V (brightness) channel
    else:
        raise ValueError(f"Unknown colorspace: {colorspace}")


# Utility functions that don't depend on class state
def apply_morphology_operation(mask: np.ndarray, kernel_size: int, iterations: int,
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


def detect_lines_in_edges(edges: np.ndarray, method: str, **params) -> Optional[np.ndarray]:
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


def point_to_line_distance(px: float, py: float, x1: float, y1: float,
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


def apply_brightness_threshold(channel: np.ndarray, threshold: int,
                               use_adaptive: bool = False, use_clahe: bool = False,
                               use_background_norm: bool = False, roi_top_ratio: float = 0.0) -> np.ndarray:
    """Apply brightness threshold with optional preprocessing enhancements and ROI masking."""

    # Step 1: Background normalization (remove slow-varying illumination)
    if use_background_norm:
        channel_float = channel.astype(np.float32)
        blur = cv2.GaussianBlur(channel_float, (31, 31), 0)
        # Subtract background and normalize
        normalized = channel_float - blur
        channel = np.zeros_like(normalized, dtype=np.uint8)
        cv2.normalize(normalized, channel, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

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


def sample_line_brightness(V: np.ndarray, x1: int, y1: int, x2: int, y2: int,
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


def merge_collinear_lines(lines: Optional[np.ndarray],
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
            dist1 = point_to_line_distance(x3, y3, x1, y1, x2, y2)
            dist2 = point_to_line_distance(x4, y4, x1, y1, x2, y2)

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


def draw_lines_on_image(image: np.ndarray, lines: Optional[np.ndarray],
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


def line_matches_target(detected_line: Tuple[int, int, int, int],
                       target_line: 'TargetLine', tolerance: float) -> bool:
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
    dist1 = point_to_line_distance(dx1, dy1, tx1, ty1, tx2, ty2)
    dist2 = point_to_line_distance(dx2, dy2, tx1, ty1, tx2, ty2)

    # Also check if target line endpoints are close to detected line
    dist3 = point_to_line_distance(tx1, ty1, dx1, dy1, dx2, dy2)
    dist4 = point_to_line_distance(tx2, ty2, dx1, dy1, dx2, dy2)

    # Line matches if endpoints are close AND angle is similar
    avg_dist = (dist1 + dist2 + dist3 + dist4) / 4
    return avg_dist < tolerance


# Type definitions for the pipeline configuration and results
class ApproximateLine(TypedDict):
    x1: int
    y1: int
    x2: int
    y2: int


class TargetLine(TypedDict):
    id: int
    name: str
    approximate_line: ApproximateLine
    tolerance: int
    description: str


class TargetsData(TypedDict):
    description: str
    image: str
    image_width: int
    image_height: int
    target_lines: List[TargetLine]
    scoring_notes: Optional[str]


class PipelineConfigRequired(TypedDict):
    # Required fields
    colorspace: Literal['gray', 'hsv']
    brightness_threshold: int
    morph_kernel_size: int
    morph_iterations: int
    use_overlay: bool
    edge_method: Literal['canny', 'sobel']
    line_method: Literal['hough', 'lsd']


class PipelineConfig(PipelineConfigRequired, total=False):
    # Optional preprocessing fields
    use_adaptive: bool
    use_clahe: bool
    use_background_norm: bool
    roi_top_ratio: float
    use_closing: bool
    merge_lines: bool
    
    # Optional edge detection parameters
    canny_low: int
    canny_high: int
    blur_kernel: int
    sobel_threshold: int
    
    # Optional line detection parameters
    hough_threshold: int
    min_line_length: int
    max_line_gap: int


class MetricsResult(TypedDict, total=False):
    count: int
    matched_targets: int
    matched_target_ids: List[int]
    precision: float
    recall: float
    f1_score: float
    score: float
    matched_line_indices: set
    qualities: List[float]
    fp_bad: int
    fp_soft: int
    config: PipelineConfig


def detect_led_lines(channel: np.ndarray, config: PipelineConfig, original_color: np.ndarray, 
                    brightness_channel: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], MetricsResult]:
    """
    Detect LED lines from a pre-processed channel.
    
    Args:
        channel: Pre-selected image channel (grayscale or HSV V-channel)
        config: Configuration dictionary
        original_color: Original color image for overlay creation
        brightness_channel: Channel to use for quality evaluation (e.g., HSV V-channel)
        
    Returns:
        Tuple of (detected_lines, metrics)
        detected_lines: numpy array of lines in format [[x1, y1, x2, y2], ...] or None
        metrics: Dictionary with detection metrics and quality scores
    """
    mask = apply_brightness_threshold(
        channel,
        config['brightness_threshold'],
        use_adaptive=config.get('use_adaptive', False),
        use_clahe=config.get('use_clahe', False),
        use_background_norm=config.get('use_background_norm', False),
        roi_top_ratio=config.get('roi_top_ratio', 0.0)
    )
    
    mask = apply_morphology_operation(
        mask,
        config['morph_kernel_size'],
        config['morph_iterations'],
        use_closing=config.get('use_closing', False)
    )
    
    processed = create_overlay(mask, config['use_overlay'], original_color)
    
    edges = detect_edges(
        processed,
        config['edge_method'],
        canny_low=config.get('canny_low', 50),
        canny_high=config.get('canny_high', 150),
        blur_kernel=config.get('blur_kernel', 0),
        sobel_threshold=config.get('sobel_threshold', 50)
    )
    
    lines = detect_lines_in_edges(
        edges,
        config['line_method'],
        hough_threshold=config.get('hough_threshold', 100),
        min_line_length=config.get('min_line_length', 100),
        max_line_gap=config.get('max_line_gap', 10)
    )
    
    if config.get('merge_lines', False):
        lines = merge_collinear_lines(lines)
    
    # Evaluate line quality (intrinsic properties only, no target matching)
    quality_metrics = evaluate_line_quality(lines, channel=brightness_channel)
    
    # Convert to MetricsResult format for compatibility
    metrics: MetricsResult = {
        'count': quality_metrics['count'],
        'matched_targets': 0,  # No target matching in this simplified interface
        'matched_target_ids': [],
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
        'score': quality_metrics['quality_score'] * 100,  # Scale quality score
        'matched_line_indices': set(),
        'qualities': quality_metrics['qualities']
    }
    
    return lines, metrics


def compute_line_quality(channel: np.ndarray, line: np.ndarray, bright_thr: int = 180) -> float:
    """
    Compute a quality score [0,1] for a detected line based on brightness profile.

    High quality lines (LED strips) should be:
    - Very bright along their length
    - Consistent brightness (low std)
    - Brighter than surrounding background

    Args:
        channel: Brightness channel (e.g., HSV V channel)
        line: Line coordinates [x1, y1, x2, y2]
        bright_thr: Threshold for "bright" pixels

    Returns:
        Quality score in [0, 1] where 1 = perfect LED candidate
    """
    x1, y1, x2, y2 = line[0]

    line_vals, bg_vals = sample_line_brightness(channel, x1, y1, x2, y2)

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


def detect_edges(image: np.ndarray, method: str, **params) -> np.ndarray:
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


def compute_lines_quality(channel: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Compute quality scores for all lines."""
    qualities = []
    for line in lines:
        q = compute_line_quality(channel, line)
        qualities.append(q)
    return np.array(qualities)


def evaluate_line_quality(lines: Optional[np.ndarray], channel: Optional[np.ndarray] = None, 
                           qualities: Optional[np.ndarray] = None, q_good: float = 0.6, q_bad: float = 0.3) -> Dict:
    """
    Evaluate lines based on their intrinsic quality properties.

    Args:
        lines: Detected lines
        channel: Brightness channel for computing qualities (e.g., HSV V channel)
        qualities: Quality scores for each line [0,1], or None to compute them
        q_good: Threshold for "good" LED candidates (>=)
        q_bad: Threshold for "bad" lines (<=)

    Returns:
        Dictionary with line quality metrics (count, qualities, quality distribution)
    """
    if lines is None or len(lines) == 0:
        return {
            'count': 0,
            'qualities': [],
            'good_lines': 0,
            'bad_lines': 0,
            'medium_lines': 0,
            'quality_score': 0.0
        }

    # Compute qualities if not provided
    if qualities is None:
        if channel is None:
            qualities = np.zeros(len(lines))  # Return zero quality scores if no channel data
        else:
            qualities = compute_lines_quality(channel, lines)

    # Classify lines by quality
    good_lines = int((qualities >= q_good).sum())
    bad_lines = int((qualities <= q_bad).sum())
    medium_lines = len(lines) - good_lines - bad_lines

    # Overall quality score based on distribution
    quality_score = float(qualities.mean()) if len(qualities) > 0 else 0.0

    return {
        'count': len(lines),
        'qualities': qualities.tolist(),
        'good_lines': good_lines,
        'bad_lines': bad_lines,
        'medium_lines': medium_lines,
        'quality_score': quality_score
    }


def evaluate_target_matching(lines: Optional[np.ndarray], targets: Optional[TargetsData] = None,
                             qualities: Optional[np.ndarray] = None, q_good: float = 0.6, q_bad: float = 0.3) -> MetricsResult:
    """
    Evaluate how well detected lines match target lines.

    Args:
        lines: Detected lines
        targets: Target lines data structure (optional)
        qualities: Quality scores for each line [0,1], required for target matching
        q_good: Threshold for "good" LED candidates (>=)
        q_bad: Threshold for "bad" lines (<=)

    Returns:
        MetricsResult with precision, recall, F1, and matching statistics
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

    # If no targets provided, fall back to simple counting
    if not targets:
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

    # Ensure we have qualities for target matching
    if qualities is None:
        qualities = np.zeros(len(lines))

    target_lines = targets.get('target_lines', [])
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
            if line_matches_target((x1, y1, x2, y2), target, tolerance):
                # Only accept match if quality is good
                if qualities[idx] >= q_good:
                    matched_targets.add(target_id)
                    matched_target_ids.append(target_id)
                    matched_line_indices.add(idx)
                    break  # Move to next target once matched

    # Classify false positives by quality
    fp_bad = 0
    fp_soft = 0
    for idx in range(len(lines)):
        if idx not in matched_line_indices:
            q = qualities[idx]
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
        'qualities': qualities.tolist()
    }


def create_overlay(mask: np.ndarray, use_overlay: bool, image: Optional[np.ndarray] = None) -> np.ndarray:
    """Create overlay of mask on original image or return mask."""
    if use_overlay and image is not None:
        return cv2.bitwise_and(image, image, mask=mask)
    return mask


class LineDetectionPipeline:
    """
    A comprehensive line detection pipeline for detecting LED strips.

    The pipeline detects bright LED lines through:
    1. Brightness thresholding and filtering
    2. Edge detection (Canny, Sobel)
    3. Line detection (Hough Transform, LSD)
    
    This class is designed to be reusable for rover robots that need to find LED lines in images.
    """

    def __init__(self, image: Optional[np.ndarray] = None, targets: Optional[TargetsData] = None):
        """
        Initialize the pipeline.
        
        Args:
            image: Input image as numpy array (BGR format). Can be set later using set_image().
            targets: Optional target lines for evaluation (loaded from JSON)
        """
        self.original_color = None
        self.gray = None
        self.hsv = None
        self.targets = targets
        
        if image is not None:
            self.set_image(image)

    def set_image(self, image: np.ndarray):
        """
        Set the input image for processing.
        
        Args:
            image: Input image as numpy array (BGR format)
        """
        self.original_color = image.copy()
        # Pre-compute color space conversions
        self.gray = cv2.cvtColor(self.original_color, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.original_color, cv2.COLOR_BGR2HSV)

    def load_targets(self, targets_path: str):
        """
        Load target lines from JSON file.
        
        Args:
            targets_path: Path to the targets JSON file
        """
        with open(targets_path, 'r') as f:
            self.targets = json.load(f)

    def process_configuration(self, config: PipelineConfig) -> Tuple[MetricsResult, List[Tuple[str, np.ndarray]]]:
        """
        Process a single configuration and return results.
        
        This method reuses detect_led_lines for the core image processing pipeline,
        then adds target matching evaluation on top of the quality-based evaluation.
        Generates visualization steps for the first few stages, then delegates to detect_led_lines.
        
        Args:
            config: Configuration dictionary containing all pipeline parameters
            
        Returns:
            Tuple of (metrics_dict, processing_steps_list)
        """
        if self.original_color is None:
            raise ValueError("No image loaded. Call set_image() first.")
        assert self.gray is not None and self.hsv is not None, "Color spaces not initialized."
            
        # Generate key visualization steps for debugging/research purposes
        steps = []
        
        # Step 0: Original image
        steps.append(('0_original', self.original_color))
        
        # Step 1: Show the selected channel
        channel = select_channel(self.gray, self.hsv, config['colorspace'])
        steps.append(('1_channel', channel))

        # Use detect_led_lines for the core line detection pipeline (avoiding duplication)
        brightness_channel = self.hsv[:, :, 2] if self.hsv is not None else None
        lines, quality_metrics = detect_led_lines(channel, config, self.original_color, brightness_channel)
        
        # Extract quality scores from the detect_led_lines result and convert to numpy array
        qualities_list = quality_metrics.get('qualities')
        qualities = np.array(qualities_list) if qualities_list is not None else None
        
        # Add target matching evaluation (the only thing this method adds over detect_led_lines)
        target_metrics = evaluate_target_matching(lines, targets=self.targets, qualities=qualities)
        
        # Use target metrics as the primary result (includes target matching info)
        metrics = target_metrics
        
        # Final step: Draw lines (color-coded by quality and match status)
        matched_indices = metrics.get('matched_line_indices', set())
        result, _ = draw_lines_on_image(self.original_color, lines, matched_indices, qualities=qualities)
        steps.append(('2_final_lines', result))

        metrics['config'] = config

        return metrics, steps