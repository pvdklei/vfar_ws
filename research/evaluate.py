import os
import cv2
import argparse
import numpy as np
from typing import List, Tuple, Optional, TypedDict, NotRequired, Literal
from itertools import product
from tqdm import tqdm
import json
import shutil

from pipeline import (
    PipelineConfig,
    detect_led_lines,
)

SELECTED_PARAM_NAMES: List["PipelineConfigKey"] = [
    "p3_edge_canny_high",
    "p3_skeleton_min_line_length",
    "p4_fusion_min_overlap",
    "p6_filter_prob_high_threshold",
]
SELECTED_PHASES: List["PipelinePhase"] = []

def optimal_config() -> PipelineConfig:
    """
    Return the best configuration found so far.

    This represents the current best-known parameter values based on
    evaluation results. Update these values as better configurations are found.

    Returns:
        Dictionary with optimal parameter values
    """

    return {
        "p1_led_map_brightness_channel": "gray",
        "p1_led_map_bg_normalize": False,
        "p3_edge_colorspace": "hsv",
        "p2_roi_mask_threshold": 200,
        "p1_led_map_use_clahe": False,
        "p2_roi_mask_use_adaptive": False,
        "p2_roi_mask_top_ratio": 0.5,
        "p2_roi_mask_morph_kernel_size": 5,
        "p2_roi_mask_morph_iterations": 1,
        "p2_roi_mask_use_closing": True,
        "p3_edge_method": "canny",
        "p3_edge_line_method": "hough",
        "p5_merge_enabled": False,
        # LED probability filtering
        "p6_filter_by_led_probability": True,
        "p6_filter_min_led_probability": 0.4,
        "p6_filter_max_lines": None,
        # Edge preprocessing
        "p3_edge_preprocess_dilate": 0,
        "p3_edge_preprocess_erode": 0,
        # LED probability evaluation
        "p6_filter_prob_high_threshold": 0.42,
        "p6_filter_prob_low_threshold": 0.1,
        "p6_filter_bright_threshold": 170,
        "p6_filter_line_sampling_width": 4,
        "p6_filter_line_sampling_num_samples": 100,
        # Line merging
        "p5_merge_angle_threshold": 7.0,
        "p5_merge_distance_threshold": 20.0,
        # Method-specific parameters
        "p3_edge_canny_low": 18,
        "p3_edge_canny_high": 120,
        "p3_edge_blur_kernel": 3,
        "p3_edge_sobel_threshold": 50,
        "p3_edge_hough_threshold": 35,
        "p3_edge_min_line_length": 48,
        "p3_edge_max_line_gap": 80,
        # LED map improvement
        "p1_led_map_gamma": 0.0,
        "p1_led_map_use_color_weighting": False,
        "p1_led_map_color_mode": "white",
        "p1_led_map_color_hue": 0.0,
        "p1_led_map_color_hue_tolerance": 15.0,
        "p1_led_map_max_white_saturation": 60,
        "p1_led_map_suppress_large_blobs": False,
        "p1_led_map_max_blob_area": 10000,
        # Branch selection
        "p4_fusion_branches": ["edge", "skeleton", "components"],
        # Edge branch
        "p3_edge_pair_angle_threshold": 3.0,
        "p3_edge_max_strip_width": 15.0,
        "p3_edge_min_pair_overlap": 0.7,
        # Region branches
        "p3_skeleton_threshold": 155,
        "p3_skeleton_morph_kernel": 3,
        "p3_skeleton_morph_iterations": 1,
        "p3_components_min_area": 2000,
        "p3_components_min_aspect_ratio": 3.0,
        "p3_skeleton_hough_threshold": 28,
        "p3_skeleton_min_line_length": 50,
        "p3_skeleton_max_line_gap": 30,
        # Fusion
        "p4_fusion_angle_tolerance": 7.0,
        "p4_fusion_distance_tolerance": 7.0,
        "p4_fusion_min_overlap": 0.5,
    }


# Pipeline phase definitions for incremental optimization
PipelinePhase = Literal[
    "p1_led_map",  # Phase 1: LED Map Creation
    "p2_roi_mask",  # Phase 2: ROI Masking
    "p3_edge",  # Phase 3a: Edge Branch Line Detection
    "p3_skeleton",  # Phase 3b: Skeleton Branch Line Detection
    "p3_components",  # Phase 3c: Components Branch Line Detection
    "p4_fusion",  # Phase 4: Branch Fusion
    "p5_merge",  # Phase 5: Line Merging
    "p6_filter",  # Phase 6: Evaluation & Filtering
]


# All valid pipeline configuration parameter keys
PipelineConfigKey = Literal[
    # Phase 1: LED Map Creation
    "p1_led_map_brightness_channel",
    "p1_led_map_bg_normalize",
    "p1_led_map_use_clahe",
    "p1_led_map_gamma",
    "p1_led_map_use_color_weighting",
    "p1_led_map_color_mode",
    "p1_led_map_color_hue",
    "p1_led_map_color_hue_tolerance",
    "p1_led_map_max_white_saturation",
    "p1_led_map_suppress_large_blobs",
    "p1_led_map_max_blob_area",
    # Phase 2: ROI Masking
    "p2_roi_mask_threshold",
    "p2_roi_mask_use_adaptive",
    "p2_roi_mask_top_ratio",
    "p2_roi_mask_morph_kernel_size",
    "p2_roi_mask_morph_iterations",
    "p2_roi_mask_use_closing",
    # Phase 3: Edge Detection
    "p3_edge_colorspace",
    "p3_edge_method",
    "p3_edge_canny_low",
    "p3_edge_canny_high",
    "p3_edge_blur_kernel",
    "p3_edge_sobel_threshold",
    "p3_edge_line_method",
    "p3_edge_hough_threshold",
    "p3_edge_min_line_length",
    "p3_edge_max_line_gap",
    "p3_edge_preprocess_dilate",
    "p3_edge_preprocess_erode",
    "p3_edge_pair_angle_threshold",
    "p3_edge_max_strip_width",
    "p3_edge_min_pair_overlap",
    # Phase 3: Skeleton Branch
    "p3_skeleton_threshold",
    "p3_skeleton_morph_kernel",
    "p3_skeleton_morph_iterations",
    "p3_skeleton_hough_threshold",
    "p3_skeleton_min_line_length",
    "p3_skeleton_max_line_gap",
    # Phase 3: Components Branch
    "p3_components_min_area",
    "p3_components_min_aspect_ratio",
    # Phase 4: Branch Fusion
    "p4_fusion_branches",
    "p4_fusion_angle_tolerance",
    "p4_fusion_distance_tolerance",
    "p4_fusion_min_overlap",
    # Phase 5: Line Merging
    "p5_merge_enabled",
    "p5_merge_angle_threshold",
    "p5_merge_distance_threshold",
    # Phase 6: Filtering
    "p6_filter_by_led_probability",
    "p6_filter_min_led_probability",
    "p6_filter_max_lines",
    "p6_filter_prob_high_threshold",
    "p6_filter_prob_low_threshold",
    "p6_filter_bright_threshold",
    "p6_filter_line_sampling_width",
    "p6_filter_line_sampling_num_samples",
]


# Mapping of phases to their parameters (using current parameter names)
PHASE_PARAMS: dict[PipelinePhase, List[PipelineConfigKey]] = {
    "p1_led_map": [
        "p1_led_map_brightness_channel",
        "p1_led_map_bg_normalize",
        "p1_led_map_use_clahe",
        "p1_led_map_gamma",
        "p1_led_map_use_color_weighting",
        "p1_led_map_color_mode",
        "p1_led_map_color_hue",
        "p1_led_map_color_hue_tolerance",
        "p1_led_map_max_white_saturation",
        "p1_led_map_suppress_large_blobs",
        "p1_led_map_max_blob_area",
    ],
    "p2_roi_mask": [
        "p2_roi_mask_threshold",
        "p2_roi_mask_use_adaptive",
        "p2_roi_mask_top_ratio",
        "p2_roi_mask_morph_kernel_size",
        "p2_roi_mask_morph_iterations",
        "p2_roi_mask_use_closing",
    ],
    "p3_edge": [
        "p3_edge_colorspace",
        "p3_edge_method",
        "p3_edge_canny_low",
        "p3_edge_canny_high",
        "p3_edge_blur_kernel",
        "p3_edge_sobel_threshold",
        "p3_edge_line_method",
        "p3_edge_hough_threshold",
        "p3_edge_min_line_length",
        "p3_edge_max_line_gap",
        "p3_edge_preprocess_dilate",
        "p3_edge_preprocess_erode",
        "p3_edge_pair_angle_threshold",
        "p3_edge_max_strip_width",
        "p3_edge_min_pair_overlap",
    ],
    "p3_skeleton": [
        "p3_skeleton_threshold",
        "p3_skeleton_morph_kernel",
        "p3_skeleton_morph_iterations",
        "p3_skeleton_hough_threshold",
        "p3_skeleton_min_line_length",
        "p3_skeleton_max_line_gap",
    ],
    "p3_components": [
        "p3_skeleton_threshold",
        "p3_skeleton_morph_kernel",
        "p3_skeleton_morph_iterations",
        "p3_components_min_area",
        "p3_components_min_aspect_ratio",
    ],
    "p4_fusion": [
        "p4_fusion_branches",
        "p4_fusion_angle_tolerance",
        "p4_fusion_distance_tolerance",
        "p4_fusion_min_overlap",
    ],
    "p5_merge": [
        "p5_merge_enabled",
        "p5_merge_angle_threshold",
        "p5_merge_distance_threshold",
    ],
    "p6_filter": [
        "p6_filter_by_led_probability",
        "p6_filter_min_led_probability",
        "p6_filter_max_lines",
        "p6_filter_prob_high_threshold",
        "p6_filter_prob_low_threshold",
        "p6_filter_bright_threshold",
        "p6_filter_line_sampling_width",
        "p6_filter_line_sampling_num_samples",
    ],
}


def search_params() -> List[PipelineConfigKey]:
    """
    Return list of parameter keys to search over.

    Only parameters listed here will be searched using the searchspace.
    All other parameters will use values from optimal_config().

    This function supports incremental optimization by allowing you to select
    specific pipeline phases to optimize. This enables a "gradient descent" style
    approach where you can:
    1. Optimize Phase 1 (LED map creation)
    2. Update optimal_config() with best Phase 1 params
    3. Optimize Phase 2 (ROI masking) while using optimal Phase 1 params
    4. Continue incrementally through all phases

    Args:
        selected_phases: List of phases to optimize. If None, uses hardcoded list below.
                        Example: ["p1_led_map", "p2_roi_mask"]

    Returns:
        List of parameter keys to include in grid search
    """

    selected_phases: list[PipelinePhase] = SELECTED_PHASES
    param_names: list[PipelineConfigKey] = SELECTED_PARAM_NAMES

    # Collect all parameters for selected phases
    for phase in selected_phases:
        if phase in PHASE_PARAMS:
            param_names.extend(PHASE_PARAMS[phase])
        else:
            raise ValueError(
                f"Unknown phase: {phase}. Valid phases: {list(PHASE_PARAMS.keys())}"
            )

    # Remove duplicates while preserving order
    # (Some params like region_threshold are shared across branches)
    seen = set()
    unique_params = []
    for param in param_names:
        if param not in seen:
            seen.add(param)
            unique_params.append(param)

    return unique_params





def create_searchspace() -> dict:
    """
    Create searchspace for parameters to explore.

    Only parameters that are in search_params() will be used from here.
    All other parameters will use values from optimal_config().

    Returns:
        Dictionary containing lists of parameter values to search over.
    """
    return {
        # Phase 1: LED Map Creation
        "p1_led_map_brightness_channels": ["gray", "hsv_v", "lab_l"],
        "p1_led_map_bg_normalizes": [False, True],
        "p1_led_map_use_clahes": [False, True],
        "p1_led_map_gammas": [0.0, 1.0],
        "p1_led_map_use_color_weightings": [False, True],
        "p1_led_map_color_modes": ["white", "hue"],
        "p1_led_map_color_hues": [0.0, 60.0],
        "p1_led_map_color_hue_tolerances": [10.0, 20.0],
        "p1_led_map_max_white_saturations": [50, 80],
        "p1_led_map_suppress_large_blobs_options": [False, True],
        "p1_led_map_max_blob_areas": [5000, 20000],
        # Phase 2: ROI Masking
        "p2_roi_mask_thresholds": [195, 205],
        "p2_roi_mask_use_adaptives": [False, True],
        "p2_roi_mask_top_ratios": [0.0, 0.5],
        "p2_roi_mask_morph_kernel_sizes": [5, 7],
        "p2_roi_mask_morph_iterations": [1, 2],
        "p2_roi_mask_use_closings": [True, False],
        # Phase 3: Edge Detection
        "p3_edge_colorspaces": ["gray", "hsv"],
        "p3_edge_methods": ["canny", "sobel"],
        "p3_edge_canny_lows": [16, 20],
        "p3_edge_canny_highs": [118, 122],
        "p3_edge_blur_kernels": [3, 5],
        "p3_edge_sobel_thresholds": [40, 60],
        "p3_edge_line_methods": ["hough", "lsd"],
        "p3_edge_hough_thresholds": [33, 37],
        "p3_edge_min_line_lengths": [46, 50],
        "p3_edge_max_line_gaps": [78, 82],
        "p3_edge_preprocess_dilates": [0, 2],
        "p3_edge_preprocess_erodes": [0, 2],
        "p3_edge_pair_angle_thresholds": [2.5, 3.5],
        "p3_edge_max_strip_widths": [15.0, 25.0],
        "p3_edge_min_pair_overlaps": [0.6, 0.8],
        # Phase 3: Skeleton Branch
        "p3_skeleton_thresholds": [153, 157],
        "p3_skeleton_morph_kernels": [3, 5],
        "p3_skeleton_morph_iterations": [1, 2],
        "p3_skeleton_hough_thresholds": [26, 30],
        "p3_skeleton_min_line_lengths": [48, 52],
        "p3_skeleton_max_line_gaps": [28, 32],
        # Phase 3: Components Branch
        "p3_components_min_areas": [500, 2000],
        "p3_components_min_aspect_ratios": [2.0, 4.0],
        # Phase 4: Branch Fusion
        "p4_fusion_branches_options": [
            ["edge"],
            ["skeleton"],
            ["components"],
        ],
        "p4_fusion_angle_tolerances": [6.5, 7.5],
        "p4_fusion_distance_tolerances": [6.5, 7.5],
        "p4_fusion_min_overlaps": [0.45, 0.55],
        # Phase 5: Line Merging
        "p5_merge_enableds": [False, True],
        "p5_merge_angle_thresholds": [3.0, 7.0],
        "p5_merge_distance_thresholds": [20.0, 40.0],
        # Phase 6: Filtering
        "p6_filter_by_led_probabilities": [True, False],
        "p6_filter_min_led_probabilities": [0.3, 0.4],
        "p6_filter_max_lines_options": [None, 8, 10, 12],
        "p6_filter_prob_high_thresholds": [0.40, 0.44],
        "p6_filter_prob_low_thresholds": [0.08, 0.12],
        "p6_filter_bright_thresholds": [168, 172],
        "p6_filter_line_sampling_widths": [3, 5],
        "p6_filter_line_sampling_num_samples_options": [50, 150],
    }


class MetricsResult(TypedDict):
    count: int
    score: float
    matched_line_indices: set
    qualities: List[float]
    # Optional fields for target-based evaluation
    matched_targets: NotRequired[int]
    matched_target_ids: NotRequired[List[int]]
    precision: NotRequired[float]
    recall: NotRequired[float]
    f1_score: NotRequired[float]
    fp_bad: NotRequired[int]
    fp_soft: NotRequired[int]
    config: NotRequired[PipelineConfig]


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


class TestCase(TypedDict):
    """A single test case with an image and its targets."""

    image: np.ndarray
    image_path: str
    targets: Optional[TargetsData]
    targets_path: Optional[str]


def point_to_line_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """Calculate perpendicular distance from point to line segment."""
    line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if line_len_sq == 0:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    # Parameter t represents position along line segment (0 to 1)
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))

    # Find projection point on line
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def line_matches_target(
    detected_line: Tuple[int, int, int, int], target_line: TargetLine, tolerance: float
) -> bool:
    """Check if a detected line matches a target line within tolerance, including angle."""
    dx1, dy1, dx2, dy2 = detected_line
    tx1 = target_line["approximate_line"]["x1"]
    ty1 = target_line["approximate_line"]["y1"]
    tx2 = target_line["approximate_line"]["x2"]
    ty2 = target_line["approximate_line"]["y2"]

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


def evaluate_target_matching(
    lines: Optional[np.ndarray],
    targets: Optional[TargetsData] = None,
    qualities: Optional[np.ndarray] = None,
    q_good: float = 0.6,
    q_bad: float = 0.3,
) -> MetricsResult:
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
            "count": 0,
            "matched_targets": 0,
            "matched_target_ids": [],
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "score": 0,
            "matched_line_indices": set(),
            "qualities": [],
        }

    # If no targets provided, fall back to simple counting
    if not targets:
        return {
            "count": len(lines),
            "matched_targets": 0,
            "matched_target_ids": [],
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "score": len(lines) * 10,
            "matched_line_indices": set(),
            "qualities": qualities.tolist() if qualities is not None else [],
        }

    # Ensure we have qualities for target matching
    if qualities is None:
        qualities = np.zeros(len(lines))

    target_lines = targets.get("target_lines", [])
    matched_targets = set()
    matched_target_ids = []
    matched_line_indices = set()

    # For each target, find if any detected line matches it
    # Only count as TP if line quality >= q_good
    for target in target_lines:
        target_id = target["id"]
        tolerance = target.get("tolerance", 20)

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
    fn = num_targets - tp  # False negatives

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
        "count": num_detected,
        "matched_targets": tp,
        "matched_target_ids": matched_target_ids,
        "precision": precision,
        "recall": recall,
        "f1_score": f_beta,  # Using F-beta score (precision-weighted)
        "fp_bad": fp_bad,
        "fp_soft": fp_soft,
        "score": score,
        "matched_line_indices": matched_line_indices,
        "qualities": qualities.tolist(),
    }


def draw_lines_with_probabilities(
    image: np.ndarray,
    lines: Optional[np.ndarray],
    probabilities: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Draw lines on image with probability-based color coding.

    Color scheme (based on LED probability):
    - Red (0, 0, 255): Low probability (0.0-0.3)
    - Yellow (0, 255, 255): Medium probability (0.3-0.6)
    - Green (0, 255, 0): High probability (0.6-1.0)

    Uses interpolation for smooth color transitions.
    """
    result = image.copy()
    if lines is None or len(lines) == 0:
        return result

    for idx, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]

        # Get probability for this line
        if probabilities is not None and idx < len(probabilities):
            prob = probabilities[idx]
        else:
            prob = 0.5  # Default to medium if no probability

        # Color interpolation based on probability
        if prob <= 0.3:
            # Low probability: Red
            color = (0, 0, 255)
        elif prob <= 0.6:
            # Medium probability: interpolate Red -> Yellow
            t = (prob - 0.3) / 0.3  # 0 to 1
            color = (0, int(255 * t), 255)
        else:
            # High probability: interpolate Yellow -> Green
            t = (prob - 0.6) / 0.4  # 0 to 1
            color = (0, 255, int(255 * (1 - t)))

        cv2.line(result, (x1, y1), (x2, y2), color, 2)

    return result


def draw_lines_on_image(
    image: np.ndarray,
    lines: Optional[np.ndarray],
    matched_targets: Optional[set] = None,
    qualities: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
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
                        color = (0, 0, 255)  # Red: bad quality
                    elif q < 0.6:
                        color = (0, 255, 255)  # Yellow: medium quality
                    else:
                        color = (0, 165, 255)  # Orange: high quality but unmatched
                else:
                    color = (0, 0, 255)  # Default red for unmatched

            cv2.line(result, (x1, y1), (x2, y2), color, 2)
            count += 1
    return result, count


def evaluate_configuration(
    image: np.ndarray, config: PipelineConfig, targets: Optional[TargetsData] = None
) -> Tuple[MetricsResult, List[Tuple[str, np.ndarray]]]:
    """
    Evaluate a single configuration on an image and return results with visualization steps.

    This function runs the line detection pipeline with the given configuration,
    evaluates the detected lines against target lines (if provided), and generates
    visualization steps for debugging/analysis.

    Args:
        image: Input image as numpy array (BGR format)
        config: Configuration dictionary containing all pipeline parameters
        targets: Optional target lines data for evaluation

    Returns:
        Tuple of (metrics_dict, processing_steps_list)
        - metrics_dict: Contains detection metrics including target matching
        - processing_steps_list: List of (step_name, image) tuples for visualization
    """
    # Generate key visualization steps for debugging/research purposes
    steps = []

    # Step 0: Original image
    steps.append(("step00_input__original_image", image))

    # Use detect_led_lines with the new API - it returns intermediate results
    lines, probability_result, intermediate = detect_led_lines(image, config)

    # Step 1: LED probability map
    led_map_colored = cv2.applyColorMap(intermediate.led_map, cv2.COLORMAP_HOT)
    steps.append(("step01_preprocessing__led_probability_map", led_map_colored))

    # Step 2: LED probability mask (unrefined)
    mask_unrefined_colored = cv2.cvtColor(
        intermediate.led_probability_mask_unrefined, cv2.COLOR_GRAY2BGR
    )
    steps.append(("step02_preprocessing__led_mask_unrefined", mask_unrefined_colored))

    # Step 3: LED probability mask (refined)
    mask_refined_colored = cv2.cvtColor(
        intermediate.led_probability_mask_refined, cv2.COLOR_GRAY2BGR
    )
    steps.append(("step03_led_mask_refined", mask_refined_colored))

    # ========================================================================
    # EDGE BRANCH VISUALIZATION
    # ========================================================================
    if intermediate.edge_branch is not None:
        # Step 4a: Edges (unmasked)
        edges_unmasked_colored = cv2.cvtColor(
            intermediate.edge_branch.edges_unmasked, cv2.COLOR_GRAY2BGR
        )
        steps.append(("step04a_edge_branch__edges_unmasked", edges_unmasked_colored))

        # Step 4b: Edges (masked)
        edges_masked_colored = cv2.cvtColor(
            intermediate.edge_branch.edges_masked, cv2.COLOR_GRAY2BGR
        )
        steps.append(("step04b_edge_branch__edges_masked", edges_masked_colored))

        # Step 4c: Raw Hough lines (before pairing)
        if intermediate.edge_branch.lines_raw is not None:
            lines_raw_img = image.copy()
            for line in intermediate.edge_branch.lines_raw:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_raw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
            steps.append(("step04c_edge_branch__lines_raw_hough", lines_raw_img))

        # Step 4d: Lines after pairing (centerlines computed)
        if intermediate.edge_branch.lines_after_pairing is not None:
            lines_paired_img = image.copy()
            for line in intermediate.edge_branch.lines_after_pairing:
                x1, y1, x2, y2 = line[0]
                cv2.line(
                    lines_paired_img, (x1, y1), (x2, y2), (0, 255, 255), 2
                )  # Yellow
            steps.append(("step04d_edge_branch__lines_after_pairing", lines_paired_img))

        # Step 4e: Lines after merging (final edge branch output)
        if intermediate.edge_branch.lines_after_merging is not None:
            lines_merged_edge_img = image.copy()
            for line in intermediate.edge_branch.lines_after_merging:
                x1, y1, x2, y2 = line[0]
                cv2.line(
                    lines_merged_edge_img, (x1, y1), (x2, y2), (0, 165, 255), 2
                )  # Orange
            steps.append(("step04e_edge_branch__lines_final", lines_merged_edge_img))

    # ========================================================================
    # SKELETON BRANCH VISUALIZATION
    # ========================================================================
    if intermediate.skeleton_branch is not None:
        # Step 5a: Skeleton branch region mask
        skeleton_mask_colored = cv2.cvtColor(
            intermediate.skeleton_branch.region_mask, cv2.COLOR_GRAY2BGR
        )
        steps.append(("step05a_skeleton_branch__mask", skeleton_mask_colored))

        # Step 5b: Skeleton (thinned centerlines)
        if intermediate.skeleton_branch.skeleton is not None:
            skeleton_colored = cv2.cvtColor(
                intermediate.skeleton_branch.skeleton, cv2.COLOR_GRAY2BGR
            )
            steps.append(("step05b_skeleton_branch__skeleton", skeleton_colored))

        # Step 5c: Skeleton mask with lines overlaid
        if intermediate.skeleton_branch.lines_detected is not None:
            mask_with_lines = cv2.cvtColor(
                intermediate.skeleton_branch.region_mask, cv2.COLOR_GRAY2BGR
            )
            for line in intermediate.skeleton_branch.lines_detected:
                x1, y1, x2, y2 = line[0]
                cv2.line(
                    mask_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2
                )  # Green on mask
            steps.append(("step05c_skeleton_branch__mask_with_lines", mask_with_lines))

        # Step 5d: Lines detected by skeleton branch (on original image)
        if intermediate.skeleton_branch.lines_detected is not None:
            lines_skeleton_img = image.copy()
            for line in intermediate.skeleton_branch.lines_detected:
                x1, y1, x2, y2 = line[0]
                cv2.line(
                    lines_skeleton_img, (x1, y1), (x2, y2), (255, 0, 255), 2
                )  # Magenta
            steps.append(
                ("step05d_skeleton_branch__lines_on_original", lines_skeleton_img)
            )

    # ========================================================================
    # COMPONENTS BRANCH VISUALIZATION
    # ========================================================================
    if intermediate.components_branch is not None:
        # Step 5e: Components branch region mask
        components_mask_colored = cv2.cvtColor(
            intermediate.components_branch.region_mask, cv2.COLOR_GRAY2BGR
        )
        steps.append(("step05e_components_branch__mask", components_mask_colored))

        # Step 5f: Components mask with lines overlaid
        if intermediate.components_branch.lines_detected is not None:
            mask_with_lines = cv2.cvtColor(
                intermediate.components_branch.region_mask, cv2.COLOR_GRAY2BGR
            )
            for line in intermediate.components_branch.lines_detected:
                x1, y1, x2, y2 = line[0]
                cv2.line(
                    mask_with_lines, (x1, y1), (x2, y2), (0, 255, 255), 2
                )  # Cyan on mask
            steps.append(
                ("step05f_components_branch__mask_with_lines", mask_with_lines)
            )

        # Step 5g: Lines detected by components branch (on original image)
        if intermediate.components_branch.lines_detected is not None:
            lines_components_img = image.copy()
            for line in intermediate.components_branch.lines_detected:
                x1, y1, x2, y2 = line[0]
                cv2.line(
                    lines_components_img, (x1, y1), (x2, y2), (0, 165, 255), 2
                )  # Orange
            steps.append(
                ("step05g_components_branch__lines_on_original", lines_components_img)
            )

    # ========================================================================
    # FUSION / COMBINED RESULTS
    # ========================================================================
    # Step 6: Combined lines from both branches (after fusion if mode='both')
    lines_detected_img = draw_lines_with_probabilities(
        image, intermediate.lines_detected, intermediate.lines_detected_probabilities
    )
    steps.append(("step06_combined__lines_after_fusion", lines_detected_img))

    # Step 7: Lines merged (with probabilities, if merging is enabled)
    if config["p5_merge_enabled"]:
        lines_merged_img = draw_lines_with_probabilities(
            image, intermediate.lines_merged, intermediate.lines_merged_probabilities
        )
        steps.append(("step07_combined__lines_after_collinear_merge", lines_merged_img))

    # Step 8: Lines final (with probabilities, after filtering)
    lines_final_img = draw_lines_with_probabilities(
        image, intermediate.lines_final, intermediate.lines_final_probabilities
    )
    steps.append(("step08_final__lines_after_probability_filter", lines_final_img))

    # Extract LED probability scores from the result and convert to numpy array
    probabilities_list = probability_result["led_probabilities"]
    qualities = np.array(probabilities_list) if probabilities_list else None

    # Add target matching evaluation
    target_metrics = evaluate_target_matching(
        lines, targets=targets, qualities=qualities
    )

    # Use target metrics as the primary result (includes target matching info)
    metrics = target_metrics

    # Final step: Draw lines with target matching (color-coded by quality and match status)
    matched_indices = metrics.get("matched_line_indices", set())
    result, _ = draw_lines_on_image(image, lines, matched_indices, qualities=qualities)
    steps.append(("step09_final__lines_with_target_matching", result))

    metrics["config"] = config

    return metrics, steps


def evaluate_configuration_multi_image(
    test_cases: List[TestCase],
    config: PipelineConfig,
) -> Tuple[MetricsResult, List[Tuple[str, str, List[Tuple[str, np.ndarray]]]]]:
    """
    Evaluate a single configuration across multiple test cases (images + targets).

    Args:
        test_cases: List of test cases, each containing an image and optional targets
        config: Configuration dictionary containing all pipeline parameters

    Returns:
        Tuple of (aggregated_metrics, per_image_results)
        - aggregated_metrics: Metrics aggregated across all test cases
        - per_image_results: List of (image_path, targets_path, steps) for each test case
    """
    per_image_results = []
    all_metrics = []

    for test_case in test_cases:
        # Evaluate single image
        metrics, steps = evaluate_configuration(
            test_case["image"], config, test_case["targets"]
        )
        all_metrics.append(metrics)
        per_image_results.append(
            (test_case["image_path"], test_case.get("targets_path", ""), steps)
        )

    # Aggregate metrics across all images
    aggregated_metrics = _aggregate_metrics(all_metrics)
    aggregated_metrics["config"] = config

    return aggregated_metrics, per_image_results


def _aggregate_metrics(metrics_list: List[MetricsResult]) -> MetricsResult:
    """
    Aggregate metrics from multiple images into a single result.

    Aggregation strategy:
    - count: mean across images
    - matched_targets: sum across images
    - precision, recall, f1_score: mean across images
    - score: mean across images (primary ranking metric)
    - fp_bad, fp_soft: sum across images
    """
    if not metrics_list:
        return MetricsResult(
            count=0,
            matched_targets=0,
            matched_target_ids=[],
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            score=0.0,
            matched_line_indices=set(),
            qualities=[],
            fp_bad=0,
            fp_soft=0,
        )

    n = len(metrics_list)

    # Aggregate counts and sums
    total_matched_targets = sum(m.get("matched_targets", 0) for m in metrics_list)
    total_fp_bad = sum(m.get("fp_bad", 0) for m in metrics_list)
    total_fp_soft = sum(m.get("fp_soft", 0) for m in metrics_list)

    # Aggregate means
    mean_count = sum(m["count"] for m in metrics_list) / n
    mean_precision = sum(m.get("precision", 0.0) for m in metrics_list) / n
    mean_recall = sum(m.get("recall", 0.0) for m in metrics_list) / n
    mean_f1 = sum(m.get("f1_score", 0.0) for m in metrics_list) / n
    mean_score = sum(m["score"] for m in metrics_list) / n

    # Collect all matched target IDs and qualities
    all_matched_ids = []
    all_qualities = []
    for m in metrics_list:
        all_matched_ids.extend(m.get("matched_target_ids", []))
        all_qualities.extend(m.get("qualities", []))

    return MetricsResult(
        count=int(mean_count),
        matched_targets=total_matched_targets,
        matched_target_ids=all_matched_ids,
        precision=mean_precision,
        recall=mean_recall,
        f1_score=mean_f1,
        score=mean_score,
        matched_line_indices=set(),  # Not meaningful when aggregated
        qualities=all_qualities,
        fp_bad=total_fp_bad,
        fp_soft=total_fp_soft,
    )


def create_grid_search_configs() -> List[PipelineConfig]:
    """
    Create grid search configurations using optimal_config as base
    and searching over parameters specified in search_params().

    Returns:
        List of pipeline configurations to evaluate
    """
    # Get base configuration with optimal values
    base_config = optimal_config()

    # Get list of parameter keys to search over
    params_to_search = search_params()

    # Get searchspace containing values to try for search parameters
    searchspace = create_searchspace()

    # Build mapping from param names to searchspace keys
    # e.g., "p2_roi_mask_threshold" -> "p2_roi_mask_thresholds"
    param_to_space_key: dict[PipelineConfigKey, str] = {
        # Phase 1: LED Map
        "p1_led_map_brightness_channel": "p1_led_map_brightness_channels",
        "p1_led_map_bg_normalize": "p1_led_map_bg_normalizes",
        "p1_led_map_use_clahe": "p1_led_map_use_clahes",
        "p1_led_map_gamma": "p1_led_map_gammas",
        "p1_led_map_use_color_weighting": "p1_led_map_use_color_weightings",
        "p1_led_map_color_mode": "p1_led_map_color_modes",
        "p1_led_map_color_hue": "p1_led_map_color_hues",
        "p1_led_map_color_hue_tolerance": "p1_led_map_color_hue_tolerances",
        "p1_led_map_max_white_saturation": "p1_led_map_max_white_saturations",
        "p1_led_map_suppress_large_blobs": "p1_led_map_suppress_large_blobs_options",
        "p1_led_map_max_blob_area": "p1_led_map_max_blob_areas",
        # Phase 2: ROI Mask
        "p2_roi_mask_threshold": "p2_roi_mask_thresholds",
        "p2_roi_mask_use_adaptive": "p2_roi_mask_use_adaptives",
        "p2_roi_mask_top_ratio": "p2_roi_mask_top_ratios",
        "p2_roi_mask_morph_kernel_size": "p2_roi_mask_morph_kernel_sizes",
        "p2_roi_mask_morph_iterations": "p2_roi_mask_morph_iterations",
        "p2_roi_mask_use_closing": "p2_roi_mask_use_closings",
        # Phase 3: Edge Detection
        "p3_edge_colorspace": "p3_edge_colorspaces",
        "p3_edge_method": "p3_edge_methods",
        "p3_edge_canny_low": "p3_edge_canny_lows",
        "p3_edge_canny_high": "p3_edge_canny_highs",
        "p3_edge_blur_kernel": "p3_edge_blur_kernels",
        "p3_edge_sobel_threshold": "p3_edge_sobel_thresholds",
        "p3_edge_line_method": "p3_edge_line_methods",
        "p3_edge_hough_threshold": "p3_edge_hough_thresholds",
        "p3_edge_min_line_length": "p3_edge_min_line_lengths",
        "p3_edge_max_line_gap": "p3_edge_max_line_gaps",
        "p3_edge_preprocess_dilate": "p3_edge_preprocess_dilates",
        "p3_edge_preprocess_erode": "p3_edge_preprocess_erodes",
        "p3_edge_pair_angle_threshold": "p3_edge_pair_angle_thresholds",
        "p3_edge_max_strip_width": "p3_edge_max_strip_widths",
        "p3_edge_min_pair_overlap": "p3_edge_min_pair_overlaps",
        # Phase 3: Skeleton Branch
        "p3_skeleton_threshold": "p3_skeleton_thresholds",
        "p3_skeleton_morph_kernel": "p3_skeleton_morph_kernels",
        "p3_skeleton_morph_iterations": "p3_skeleton_morph_iterations",
        "p3_skeleton_hough_threshold": "p3_skeleton_hough_thresholds",
        "p3_skeleton_min_line_length": "p3_skeleton_min_line_lengths",
        "p3_skeleton_max_line_gap": "p3_skeleton_max_line_gaps",
        # Phase 3: Components Branch
        "p3_components_min_area": "p3_components_min_areas",
        "p3_components_min_aspect_ratio": "p3_components_min_aspect_ratios",
        # Phase 4: Branch Fusion
        "p4_fusion_branches": "p4_fusion_branches_options",
        "p4_fusion_angle_tolerance": "p4_fusion_angle_tolerances",
        "p4_fusion_distance_tolerance": "p4_fusion_distance_tolerances",
        "p4_fusion_min_overlap": "p4_fusion_min_overlaps",
        # Phase 5: Line Merging
        "p5_merge_enabled": "p5_merge_enableds",
        "p5_merge_angle_threshold": "p5_merge_angle_thresholds",
        "p5_merge_distance_threshold": "p5_merge_distance_thresholds",
        # Phase 6: Filtering
        "p6_filter_by_led_probability": "p6_filter_by_led_probabilities",
        "p6_filter_min_led_probability": "p6_filter_min_led_probabilities",
        "p6_filter_max_lines": "p6_filter_max_lines_options",
        "p6_filter_prob_high_threshold": "p6_filter_prob_high_thresholds",
        "p6_filter_prob_low_threshold": "p6_filter_prob_low_thresholds",
        "p6_filter_bright_threshold": "p6_filter_bright_thresholds",
        "p6_filter_line_sampling_width": "p6_filter_line_sampling_widths",
        "p6_filter_line_sampling_num_samples": "p6_filter_line_sampling_num_samples_options",
    }

    # Extract values to search for each parameter
    search_values = {}
    for param in params_to_search:
        space_key = param_to_space_key.get(param)
        if space_key is None:
            raise ValueError(f"No mapping found for search parameter: {param}")
        if space_key not in searchspace:
            raise ValueError(f"Searchspace key not found: {space_key}")
        search_values[param] = searchspace[space_key]

    # Generate all combinations
    configs = []
    param_names = list(search_values.keys())
    param_value_lists = [search_values[param] for param in param_names]

    for value_combination in product(*param_value_lists):
        # Start with base config
        config = base_config.copy()

        # Update with searched parameter values
        for param_name, value in zip(param_names, value_combination):
            config[param_name] = value

        configs.append(config)

    print(f"Generated {len(configs)} configurations for grid search")
    print(f"Searching over {len(params_to_search)} parameters: {params_to_search}")
    return configs


def generate_config_name(config: PipelineConfig) -> str:
    """Generate a human-readable name encoding all important config parameters."""
    # Build preprocessing flags
    flags = []
    if config["p1_led_map_bg_normalize"]:
        flags.append("bg")
    if config["p1_led_map_use_clahe"]:
        flags.append("cl")
    if config["p2_roi_mask_use_adaptive"]:
        flags.append("ad")
    if config["p3_edge_blur_kernel"] > 0:
        flags.append(f"b{config['p3_edge_blur_kernel']}")
    if config["p5_merge_enabled"]:
        flags.append("mg")
    flags_str = "-".join(flags) if flags else "plain"

    parts = [
        config["p3_edge_colorspace"],
        f"thr{config['p2_roi_mask_threshold']}",
        f"k{config['p2_roi_mask_morph_kernel_size']}i{config['p2_roi_mask_morph_iterations']}",
        config["p3_edge_method"],
        config["p3_edge_line_method"],
    ]

    # Add edge/line detection params
    if config["p3_edge_canny_low"]:
        parts.append(f"cl{config['p3_edge_canny_low']}")
    if config["p3_edge_canny_high"]:
        parts.append(f"ch{config['p3_edge_canny_high']}")
    if config["p3_edge_hough_threshold"]:
        parts.append(f"h{config['p3_edge_hough_threshold']}")
    if config["p3_edge_min_line_length"]:
        parts.append(f"ml{config['p3_edge_min_line_length']}")

    # Add ROI if used
    if config["p2_roi_mask_top_ratio"] > 0:
        parts.append(f"roi{int(config['p2_roi_mask_top_ratio']*100)}")

    # Add preprocessing flags at the end
    parts.append(flags_str)

    return "_".join(str(p) for p in parts if p)


def save_results(
    output_folder: str,
    config_id: int,
    config: PipelineConfig,
    metrics: MetricsResult,
    per_image_results: List[Tuple[str, str, List[Tuple[str, np.ndarray]]]],
):
    """Save all images for a configuration across multiple test cases.

    Args:
        output_folder: Base output folder
        config_id: Configuration ID for folder naming
        config: Pipeline configuration
        metrics: Aggregated metrics across all images
        per_image_results: List of (image_path, targets_path, steps) for each test case
    """
    # Create folder for this configuration
    config_name = generate_config_name(config)
    folder_path = os.path.join(output_folder, f"{config_id:03d}_{config_name}")
    os.makedirs(folder_path, exist_ok=True)

    # Save results for each test image in its own subfolder
    for image_path, targets_path, steps in per_image_results:
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        image_folder = os.path.join(folder_path, image_basename)
        os.makedirs(image_folder, exist_ok=True)

        # Save all intermediate steps for this image
        for step_name, image in steps:
            cv2.imwrite(os.path.join(image_folder, f"{step_name}.png"), image)

    # Save configuration and aggregated metrics in the root of the config folder
    info = {
        "config": config,
        "metrics": {
            k: (
                float(v)
                if isinstance(v, (np.floating, np.integer))
                else list(v) if isinstance(v, set) else v
            )
            for k, v in metrics.items()
            if k not in ["config", "matched_line_indices"]
        },
        "num_test_images": len(per_image_results),
        "test_images": [
            os.path.basename(img_path) for img_path, _, _ in per_image_results
        ],
    }
    with open(os.path.join(folder_path, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    return folder_path


def draw_target_lines(
    image: np.ndarray, targets: Optional[TargetsData], save_path: str
):
    """Draw target lines on original image and save visualization."""
    if not targets or image is None:
        return

    result = image.copy()
    target_lines = targets.get("target_lines", [])

    for target in target_lines:
        line_info = target["approximate_line"]
        x1, y1 = line_info["x1"], line_info["y1"]
        x2, y2 = line_info["x2"], line_info["y2"]
        target_id = target["id"]

        # Draw target line in blue
        cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Add label with ID
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(
            result,
            f"T{target_id}",
            (mid_x, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    cv2.imwrite(save_path, result)
    print(f"Target lines visualization saved to: {save_path}")


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
        "--images",
        type=str,
        default="images",
        help="Path to folder containing test images (default: images)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="targets",
        help="Path to folder containing target JSON files (default: targets). Each JSON should have the same basename as its corresponding image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("output", "linedetect"),
        help="Path to save results (default: output/linedetect)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top configurations to save (default: 10)",
    )
    parser.add_argument(
        "--preview-targets",
        action="store_true",
        help="Only generate target_lines_reference.png and exit (for quick editing)",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Only test the optimal configuration (skip grid search)",
    )
    args = parser.parse_args()

    # Clean output folder (delete everything inside it for a fresh start)
    if os.path.exists(args.output):
        print(f"Cleaning output folder: {args.output}")
        shutil.rmtree(args.output)

    # Create output folder
    os.makedirs(args.output, exist_ok=True)

    # Load all test images and their targets
    print(f"\nLoading images from: {args.images}")
    print(f"Loading targets from: {args.targets}")

    test_cases: List[TestCase] = []

    # Get all image files from images folder
    if not os.path.exists(args.images):
        print(f"Error: Images folder not found: {args.images}")
        return

    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    image_files = [
        f for f in os.listdir(args.images) if f.lower().endswith(image_extensions)
    ]

    if not image_files:
        print(f"Error: No images found in {args.images}")
        return

    print(f"Found {len(image_files)} images")

    # Load each image and its corresponding targets
    for image_file in sorted(image_files):
        image_path = os.path.join(args.images, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Could not load image: {image_file}, skipping...")
            continue

        # Try to find corresponding targets file
        basename = os.path.splitext(image_file)[0]
        targets_file = basename + ".json"
        targets_path = os.path.join(args.targets, targets_file)

        targets = None
        if os.path.exists(targets_path):
            try:
                with open(targets_path, "r") as f:
                    targets = json.load(f)
                print(f"  ✓ {image_file} -> {targets_file}")
            except Exception as e:
                print(f"  ✗ {image_file} -> {targets_file} (error: {e})")
        else:
            print(f"  ○ {image_file} -> no targets")
            targets_path = None

        test_cases.append(
            {
                "image": image,
                "image_path": image_path,
                "targets": targets,
                "targets_path": targets_path,
            }
        )

    if not test_cases:
        print("Error: No valid test cases loaded")
        return

    print(f"\nLoaded {len(test_cases)} test cases")

    # Draw target lines visualizations for all images with targets
    target_viz_folder = os.path.join(args.output, "target_visualizations")
    os.makedirs(target_viz_folder, exist_ok=True)

    for test_case in test_cases:
        if test_case["targets"]:
            basename = os.path.splitext(os.path.basename(test_case["image_path"]))[0]
            target_viz_path = os.path.join(target_viz_folder, f"{basename}_targets.png")
            draw_target_lines(test_case["image"], test_case["targets"], target_viz_path)

    # If preview mode, exit after drawing targets
    if args.preview_targets:
        print("\nPreview mode: Target visualizations complete. Exiting.")
        return

    # Generate configurations
    print("\n" + "=" * 80)
    if args.best:
        print("TESTING OPTIMAL CONFIGURATION")
        print("=" * 80)
        configs = [optimal_config()]
        print("Using best known configuration from optimal_config()")
    else:
        print("GENERATING GRID SEARCH CONFIGURATIONS")
        print("=" * 80)
        configs = create_grid_search_configs()

    # Process all configurations
    print("\n" + "=" * 80)
    print("PROCESSING CONFIGURATIONS")
    print("=" * 80)
    results = []

    with tqdm(
        total=len(configs),
        desc="Overall Progress",
        position=0,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar:
        for i, config in enumerate(configs):
            # Evaluate configuration across all test cases
            metrics, per_image_results = evaluate_configuration_multi_image(
                test_cases, config
            )

            results.append(
                {
                    "config_id": i,
                    "config": config,
                    "metrics": metrics,
                    "per_image_results": per_image_results,
                }
            )

            pbar.update(1)
            # Update the main progress bar with current best score
            if results:
                best_score = max(r["metrics"]["score"] for r in results)
                pbar.set_postfix({"best_score": f"{best_score:.1f}"})

    # Sort by score (descending)
    print("\n\nSorting results by score...")
    results.sort(key=lambda x: x["metrics"]["score"], reverse=True)

    # Save top-k results
    print("\n" + "=" * 80)
    print(f"SAVING TOP {args.top_k} CONFIGURATIONS")
    print("=" * 80)
    top_results = []

    with tqdm(
        total=args.top_k,
        desc="Saving Results",
        position=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
    ) as pbar:
        for rank, result in enumerate(results[: args.top_k]):
            folder = save_results(
                args.output,
                rank + 1,
                result["config"],
                result["metrics"],
                result["per_image_results"],
            )

            metrics_to_save = {
                "rank": rank + 1,
                "folder": folder,
                "score": result["metrics"]["score"],
                "line_count": result["metrics"]["count"],
                "config": result["config"],
            }

            # Add target-based metrics if available
            if "matched_targets" in result["metrics"]:
                metrics_to_save.update(
                    {
                        "matched_targets": result["metrics"]["matched_targets"],
                        "matched_target_ids": result["metrics"]["matched_target_ids"],
                        "precision": result["metrics"]["precision"],
                        "recall": result["metrics"]["recall"],
                        "f1_score": result["metrics"]["f1_score"],
                    }
                )

            top_results.append(metrics_to_save)

            # Update progress bar
            postfix = {"rank": rank + 1, "score": f"{result['metrics']['score']:.1f}"}
            if "matched_targets" in result["metrics"]:
                postfix["matched"] = (
                    f"{result['metrics']['matched_targets']}/{result['metrics']['count']}"
                )
            pbar.set_postfix(postfix)
            pbar.update(1)

    # Save summary
    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(top_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("TOP CONFIGURATIONS")
    print("=" * 80)
    for res in top_results[:5]:
        print(f"\nRank {res['rank']}: Score={res['score']:.1f}")
        print(f"  Lines Detected: {res['line_count']}")

        # Show target-based metrics if available
        if "matched_targets" in res:
            # Count total targets across all test cases
            total_targets = sum(
                len(tc["targets"]["target_lines"]) for tc in test_cases if tc["targets"]
            )
            print(f"  Target Matches: {res['matched_targets']}/{total_targets} targets")
            print(f"  Matched IDs: {res['matched_target_ids']}")
            print(
                f"  Precision: {res['precision']:.1%}, Recall: {res['recall']:.1%}, F1: {res['f1_score']:.3f}"
            )

        print(f"  Config: {generate_config_name(res['config'])}")
        print(f"  Folder: {res['folder']}")

    print(f"\nFull summary saved to: {summary_path}")
    print(f"Total configurations tested: {len(configs)}")
    print(f"Top {args.top_k} results saved to: {args.output}")


if __name__ == "__main__":
    main()
