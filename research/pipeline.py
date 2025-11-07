import cv2
import numpy as np
from typing import List, Tuple, Optional, TypedDict, Literal
from dataclasses import dataclass


class PipelineConfig(TypedDict):
    """Configuration for the LED line detection pipeline.

    All fields are required to ensure explicit configuration and avoid
    ambiguity about default values.

    Pipeline Overview:
    1. Create LED probability map from input image (_create_led_map)
    2. Create and refine LED probability mask (_create_led_probability_mask, _refine_led_probability_mask)
    3. Detect edges in unmasked image (_detect_edges)
    4. Apply LED probability mask to edges
    5. Detect lines in masked edges (_detect_lines_in_edges)
    6. Optionally merge collinear lines (_merge_collinear_lines)
    7. Evaluate lines by LED probability (_evaluate_lines_by_led_probability)
    8. Filter lines by LED probability (_filter_lines_by_led_probability)
    """

    # ============================================================================
    # STEP 1: LED Probability Map Creation (_create_led_map)
    # ============================================================================
    p1_led_map_brightness_channel: Literal["gray", "hsv_v", "lab_l"]
    """Which brightness channel to extract from BGR image.
    - 'gray': Standard grayscale (simple, fast)
    - 'hsv_v': HSV Value channel (good for varying lighting)
    - 'lab_l': LAB Lightness (perceptually uniform brightness)
    """

    p1_led_map_bg_normalize: bool
    """Apply background normalization using large Gaussian blur to remove
    slow-varying illumination and highlight LED strips.
    """

    p1_led_map_gamma: float
    """Gamma correction value for brightness enhancement (0.0 = disabled, >0 = enabled).
    Values < 1.0 boost bright pixels (LEDs) relative to medium/dark regions.
    Typical values: 0.5-0.8 for strong enhancement, 1.0 for no effect.
    Apply after background normalization but before CLAHE.
    """

    p1_led_map_use_clahe: bool
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance
    local contrast and make LED strips more distinct from background.
    """

    p1_led_map_use_color_weighting: bool
    """Enable color-based weighting to suppress bright regions with wrong colors.
    Helps filter out windows, walls, and other bright non-LED regions.
    """

    p1_led_map_color_mode: Literal["white", "hue"]
    """LED color mode for color weighting:
    - 'white': Prefer low-saturation bright regions (white/warm white LEDs)
    - 'hue': Prefer specific hue range (colored LEDs)
    """

    p1_led_map_color_hue: float
    """Target hue for colored LEDs (0-180 in OpenCV HSV).
    Only used when p1_led_map_color_mode='hue'.
    Examples: 0=red, 30=orange, 60=yellow, 90=green, 120=cyan, 150=blue.
    """

    p1_led_map_color_hue_tolerance: float
    """Hue tolerance for colored LED matching (degrees, e.g., 10-20).
    Only used when p1_led_map_color_mode='hue'.
    Larger values = more permissive color matching.
    """

    p1_led_map_max_white_saturation: int
    """Saturation scaling denominator (0-255) for color weighting.
    - In 'white' mode: Pixels with saturation above this are downweighted
    - In 'hue' mode: Used as denominator for saturation boost (higher saturation → higher weight)
    Typical value: 60-80 for white LEDs, may need adjustment for colored LEDs.
    Must be > 0 to avoid division by zero (will be clamped to min of 1 if invalid).
    """

    p1_led_map_suppress_large_blobs: bool
    """Remove large bright blobs (windows, whiteboards) from LED probability map.
    Uses connected components to identify and suppress large bright regions
    that are unlikely to be LED strips.
    """

    p1_led_map_max_blob_area: int
    """Maximum blob area in pixels for large blob suppression.
    Bright regions larger than this are suppressed.
    Typical value: 1-5% of image area (e.g., 10000-50000 for typical images).
    Only used when p1_led_map_suppress_large_blobs=True.
    """

    # ============================================================================
    # STEP 2: LED Probability Masking (_create_led_probability_mask)
    # ============================================================================
    p2_roi_mask_threshold: int
    """Threshold value (0-255) for binary thresholding of LED probability map.
    Pixels above this value are considered LED candidates.
    Higher values = stricter (fewer false positives, may miss dim LEDs).
    """

    p2_roi_mask_use_adaptive: bool
    """Use adaptive thresholding instead of global thresholding.
    Better for images with varying illumination across the scene.
    """

    p2_roi_mask_top_ratio: float
    """Ratio (0.0-1.0) of image height to use as Region of Interest.
    For ceiling-mounted LEDs, set to 0.5-0.6 to focus on top portion.
    0.0 = use full image (no ROI restriction).
    """

    # ============================================================================
    # STEP 2B: Morphological Mask Refinement (_refine_led_probability_mask)
    # ============================================================================
    p2_roi_mask_morph_kernel_size: int
    """Size of morphological operation kernel (e.g., 3, 5, 7).
    Larger kernels connect larger gaps but may merge separate LED strips.
    0 = disable morphological operations.
    """

    p2_roi_mask_morph_iterations: int
    """Number of morphological operation iterations.
    More iterations = stronger effect but may distort mask shape.
    0 = disable morphological operations.
    """

    p2_roi_mask_use_closing: bool
    """Use closing (dilation→erosion) instead of just dilation.
    Closing fills internal gaps while preserving overall size better.
    """

    # ============================================================================
    # STEP 3: Edge Detection (_detect_edges)
    # ============================================================================
    p3_edge_colorspace: Literal["gray", "hsv"]
    """Colorspace for edge detection channel extraction.
    - 'gray': Grayscale (standard, works well in most cases)
    - 'hsv': HSV Value channel (brightness-based edges)
    """

    p3_edge_method: Literal["canny", "sobel"]
    """Edge detection algorithm.
    - 'canny': Canny edge detector (better for clean edges, needs tuning)
    - 'sobel': Sobel gradient magnitude (simpler, more robust to noise)
    """

    p3_edge_canny_low: int
    """Low threshold for Canny edge detection (e.g., 50).
    Lower values detect more edges (higher sensitivity).
    Only used if p3_edge_method='canny'.
    """

    p3_edge_canny_high: int
    """High threshold for Canny edge detection (e.g., 150).
    Should be 2-3x canny_low. Higher values = stricter edge detection.
    Only used if p3_edge_method='canny'.
    """

    p3_edge_blur_kernel: int
    """Gaussian blur kernel size before edge detection (must be odd, e.g., 3, 5).
    Reduces noise but may blur fine edges. 0 = no blur.
    """

    p3_edge_sobel_threshold: int
    """Threshold for Sobel gradient magnitude (e.g., 50).
    Higher values = only strong edges detected.
    Only used if p3_edge_method='sobel'.
    """

    # ============================================================================
    # STEP 5: Line Detection in Edges (_detect_lines_in_edges)
    # ============================================================================
    p3_edge_line_method: Literal["hough", "lsd"]
    """Line detection algorithm.
    - 'hough': Probabilistic Hough Transform (fast, good for straight lines)
    - 'lsd': Line Segment Detector (more accurate, slower)
    """

    p3_edge_hough_threshold: int
    """Hough transform accumulator threshold (e.g., 50-100).
    Higher values require more evidence for a line (fewer false positives).
    Only used if p3_edge_line_method='hough'.
    """

    p3_edge_min_line_length: int
    """Minimum line length in pixels (e.g., 50-100).
    Shorter segments are rejected. Should match expected LED strip length.
    """

    p3_edge_max_line_gap: int
    """Maximum gap in pixels to connect line segments (e.g., 10-20).
    Useful for connecting broken LED lines due to gaps or occlusions.
    """

    p3_edge_preprocess_dilate: int
    """Dilation kernel size for connecting nearby edges before line detection.
    Useful for discontinuous LED strips. 0 = disable.
    """

    p3_edge_preprocess_erode: int
    """Erosion kernel size for thinning edges before line detection.
    Removes noise and thin edges. 0 = disable.
    """

    # ============================================================================
    # EDGE BRANCH: Paired-Edge Centerline Detection
    # ============================================================================
    p3_edge_pair_angle_threshold: float
    """Maximum angle difference for pairing parallel lines (degrees, e.g., 3.0).
    Lines with angle difference <= this value are candidates for edge pairing.
    Used in edge branch to find paired edges of LED strips.
    """

    p3_edge_max_strip_width: float
    """Maximum perpendicular distance for pairing parallel lines (pixels, e.g., 20.0).
    Lines farther apart than this won't be paired even if parallel.
    Should match expected LED strip width in the image.
    """

    p3_edge_min_pair_overlap: float
    """Minimum overlap ratio for pairing parallel lines (0-1, e.g., 0.7).
    Lines must overlap at least this much along their direction to be paired.
    Higher values = stricter pairing (only well-aligned edge pairs).
    """

    # ============================================================================
    # BRANCH SELECTION AND FUSION
    # ============================================================================
    p4_fusion_branches: List[Literal["edge", "skeleton", "components"]]
    """List of detection branches to use:
    - 'edge': Edge branch (Canny + Hough + paired-edge centerlines)
    - 'skeleton': Skeleton branch (LED map -> skeleton -> Hough)
    - 'components': Components branch (LED map -> components -> PCA)

    Multiple branches can be enabled and their results will be fused.
    Examples:
    - ['edge'] - Only edge detection
    - ['skeleton', 'components'] - Both region-based approaches
    - ['edge', 'skeleton', 'components'] - All three branches
    """

    p3_skeleton_threshold: int
    """Threshold for LED map binarization (0-255, e.g., 180).
    Pixels above this value are considered part of LED regions.
    Higher values = stricter (only brightest regions).
    """

    p3_skeleton_morph_kernel: int
    """Morphology kernel size for region cleanup (pixels, e.g., 3-7).
    Used to close gaps and smooth region boundaries.
    """

    p3_skeleton_morph_iterations: int
    """Morphology iterations for region cleanup (e.g., 1-3).
    More iterations = more aggressive cleanup.
    """

    p3_components_min_area: int
    """Minimum component area in pixels (e.g., 500-2000).
    Smaller regions are filtered out as noise.
    Only used in 'components_pca' mode.
    """

    p3_components_min_aspect_ratio: float
    """Minimum aspect ratio for elongated regions (e.g., 2.0-4.0).
    Regions with lower aspect ratio are filtered out.
    Only used in 'components_pca' mode.
    """

    p3_skeleton_hough_threshold: int
    """Hough threshold for skeleton-based detection (e.g., 30-50).
    Lower than edge Hough since skeleton is thinner.
    Only used in 'skeleton_hough' mode.
    """

    p3_skeleton_min_line_length: int
    """Minimum line length for skeleton detection (pixels, e.g., 40-60).
    Only used in 'skeleton_hough' mode.
    """

    p3_skeleton_max_line_gap: int
    """Maximum line gap for skeleton detection (pixels, e.g., 20-40).
    Only used in 'skeleton_hough' mode.
    """

    # ============================================================================
    # FUSION: Combining Edge and Region Branch Results
    # ============================================================================
    p4_fusion_angle_tolerance: float
    """Maximum angle difference for considering lines similar during fusion (degrees, e.g., 3.0-5.0).
    Lines with angle difference <= this value may be considered redundant.
    Only used when multiple branches are enabled.
    """

    p4_fusion_distance_tolerance: float
    """Maximum perpendicular distance for considering lines similar during fusion (pixels, e.g., 10.0).
    Lines farther apart than this are considered distinct.
    Only used when multiple branches are enabled.
    """

    p4_fusion_min_overlap: float
    """Minimum overlap ratio for considering lines similar during fusion (0-1, e.g., 0.5).
    Lines with lower overlap are considered distinct.
    Only used when multiple branches are enabled.
    """

    # ============================================================================
    # STEP 6: Line Merging (_merge_collinear_lines)
    # ============================================================================
    p5_merge_enabled: bool
    """Whether to merge nearby collinear line segments into single lines.
    Useful when line detector splits one LED strip into multiple segments.
    """

    # ============================================================================
    # STEP 8: Line Filtering (_filter_lines_by_led_probability)
    # ============================================================================
    p6_filter_by_led_probability: bool
    """Whether to filter detected lines based on LED probability scores.
    Removes lines unlikely to be LED strips based on LED probability analysis.
    """

    p6_filter_min_led_probability: float
    """Minimum LED probability threshold (0.0-1.0, e.g., 0.3).
    Lines with scores below this are filtered out.
    Higher values = stricter filtering (only high-confidence LED lines).
    """

    p6_filter_max_lines: Optional[int]
    """Maximum number of lines to return (e.g., 4, 6).
    If more lines pass min_led_probability, only the top max_lines with
    highest scores are kept. None = no limit (return all passing lines).
    """

    # ============================================================================
    # STEP 7: Line Evaluation (_evaluate_lines_by_led_probability)
    # ============================================================================
    p6_filter_prob_high_threshold: float
    """Threshold for classifying lines as "high probability" LED candidates (e.g., 0.6).
    Lines with LED probability scores >= this value are considered high confidence.
    Used for statistics and reporting in evaluation results.
    """

    p6_filter_prob_low_threshold: float
    """Threshold for classifying lines as "low probability" LED candidates (e.g., 0.3).
    Lines with LED probability scores <= this value are considered low confidence.
    Used for statistics and reporting in evaluation results.
    """

    p6_filter_bright_threshold: int
    """Brightness threshold (0-255) for determining if a pixel is "bright" when
    computing LED probability scores (e.g., 180). Used to calculate the ratio
    of bright pixels along each detected line. Higher values = stricter.
    """

    p6_filter_line_sampling_width: int
    """Width in pixels for perpendicular sampling when computing LED probability (e.g., 3).
    Samples background pixels this many pixels away on each side of the line
    to measure contrast between line and background.
    """

    p6_filter_line_sampling_num_samples: int
    """Number of sample points along the line for LED probability computation (e.g., 100).
    More samples give better statistics but are slower to compute.
    """

    # ============================================================================
    # STEP 6: Line Merging (_merge_collinear_lines)
    # ============================================================================
    p5_merge_angle_threshold: float
    """Maximum angle difference in degrees for merging collinear lines (e.g., 5.0).
    Lines with angle difference <= this value are candidates for merging.
    Smaller values = stricter collinearity requirement.
    """

    p5_merge_distance_threshold: float
    """Maximum perpendicular distance in pixels for merging collinear lines (e.g., 30.0).
    Lines farther apart than this distance will not be merged even if collinear.
    Smaller values = stricter proximity requirement.
    """


class LineLEDProbabilityResult(TypedDict):
    """Result from line evaluation containing LED probability metrics for detected lines."""

    count: int
    led_probabilities: List[float]
    high_prob_lines: int
    low_prob_lines: int
    medium_prob_lines: int
    mean_probability: float


@dataclass
class EdgeBranchResults:
    """Intermediate results specific to the edge-based detection branch."""
    edges_unmasked: np.ndarray
    edges_masked: np.ndarray
    lines_raw: Optional[np.ndarray]  # Before pairing/merging
    lines_after_pairing: Optional[np.ndarray]  # After paired-edge centerline detection
    lines_after_merging: Optional[np.ndarray]  # After collinear merging


@dataclass
class RegionBranchResults:
    """Intermediate results specific to the region-based detection branch."""
    region_mask: np.ndarray  # Thresholded and morphologically processed mask
    skeleton: Optional[np.ndarray]  # Skeleton (for skeleton_hough mode)
    lines_detected: Optional[np.ndarray]  # Lines from region detection


@dataclass
class PipelineIntermediateResults:
    """Container for all intermediate results from the LED line detection pipeline.

    This class stores images, masks, and line sets from each pipeline step for
    visualization and debugging purposes.
    """
    # Step 0: Input
    original_image: np.ndarray

    # Step 1: LED Probability Map
    led_map: np.ndarray

    # Step 2: LED Probability Masking
    led_probability_mask_unrefined: np.ndarray
    led_probability_mask_refined: np.ndarray

    # Branch-specific results (multiple branches can be populated)
    edge_branch: Optional[EdgeBranchResults]
    skeleton_branch: Optional[RegionBranchResults]
    components_branch: Optional[RegionBranchResults]

    # Final results (branch-agnostic)
    lines_detected: Optional[np.ndarray]  # Lines after branch processing
    lines_detected_probabilities: List[float]
    lines_merged: Optional[np.ndarray]  # Same as detected for now (kept for compatibility)
    lines_merged_probabilities: List[float]
    lines_final: Optional[np.ndarray]  # After filtering
    lines_final_probabilities: List[float]
    probability_result: LineLEDProbabilityResult


def detect_led_lines(
    image_bgr: np.ndarray, config: PipelineConfig
) -> Tuple[Optional[np.ndarray], LineLEDProbabilityResult, PipelineIntermediateResults]:
    """
    Detect LED lines from a BGR color image.

    This is the main entry point for the line detection pipeline. It handles all
    channel conversions and processing internally based on the configuration.

    Pipeline steps:
    1. Create LED probability map from input image
    2. Create and refine LED probability mask (binary mask of LED candidate regions)
    3. Detect edges in unmasked image
    4. Apply LED probability mask to edges
    5. Detect lines in masked edges
    6. Optionally merge collinear lines
    7. Evaluate all lines by LED probability (computes scores once for efficiency)
    8. Filter lines by LED probability (removes low-probability lines, limits to max_lines using pre-computed scores)

    Args:
        image_bgr: BGR color image (OpenCV format)
        config: Pipeline configuration dictionary (PipelineConfig). All parameters are required
            to ensure explicit configuration. Key parameters include:
            - p3_edge_colorspace: Colorspace for edge detection ('gray', 'hsv')
            - brightness_channel_type: Channel type for LED map ('gray', 'hsv_v', 'lab_l')
            - brightness_threshold: Threshold for LED candidate detection
            - p3_edge_method: Edge detection method ('canny', 'sobel')
            - p3_edge_line_method: Line detection method ('hough', 'lsd')
            - p6_filter_by_led_probability: Whether to filter lines by LED probability
            - p6_filter_min_led_probability: Minimum LED probability threshold
            - p6_filter_max_lines: Maximum number of lines to return (None for unlimited)
            - See PipelineConfig TypedDict for complete list of required parameters

    Returns:
        Tuple of (detected_lines, probability_result)
        - detected_lines: Array of shape (N, 1, 4) where each line is [[x1, y1, x2, y2]], or None
        - probability_result: LineLEDProbabilityResult with LED probability metrics
          (count, led_probabilities, high_prob_lines, low_prob_lines, medium_prob_lines, mean_probability)
    """
    # Create LED probability map (high values = likely LED locations)
    led_map = _create_led_map(
        image_bgr,
        config["p1_led_map_brightness_channel"],
        p1_led_map_use_clahe=config["p1_led_map_use_clahe"],
        p1_led_map_bg_normalize=config["p1_led_map_bg_normalize"],
        p1_led_map_gamma=config["p1_led_map_gamma"],
        p1_led_map_use_color_weighting=config["p1_led_map_use_color_weighting"],
        p1_led_map_color_mode=config["p1_led_map_color_mode"],
        p1_led_map_color_hue=config["p1_led_map_color_hue"],
        p1_led_map_color_hue_tolerance=config["p1_led_map_color_hue_tolerance"],
        p1_led_map_max_white_saturation=config["p1_led_map_max_white_saturation"],
        p1_led_map_suppress_large_blobs=config["p1_led_map_suppress_large_blobs"],
        p1_led_map_max_blob_area=config["p1_led_map_max_blob_area"],
    )

    # Step 1: Create LED probability mask (binary mask of potential LED regions)
    led_probability_mask_unrefined = _create_led_probability_mask(
        led_map,
        config["p2_roi_mask_threshold"],
        p2_roi_mask_use_adaptive=config["p2_roi_mask_use_adaptive"],
        p2_roi_mask_top_ratio=config["p2_roi_mask_top_ratio"],
    )

    # Step 2: Refine mask using morphological operations
    led_probability_mask_refined = _refine_led_probability_mask(
        led_probability_mask_unrefined,
        config["p2_roi_mask_morph_kernel_size"],
        config["p2_roi_mask_morph_iterations"],
        p2_roi_mask_use_closing=config["p2_roi_mask_use_closing"],
    )

    # Step 3-6: Line detection using configured branch(es)
    # Supports: edge, skeleton, components, or any combination
    edge_branch_results = None
    skeleton_branch_results = None
    components_branch_results = None
    lines_edge = None
    lines_skeleton = None
    lines_components = None
    lines_merged = None

    # Run branch(es) based on branches list
    if "edge" in config["p4_fusion_branches"]:
        # Run edge branch
        lines_edge, edge_branch_results = _detect_lines_edge_branch(
            image_bgr=image_bgr,
            led_map=led_map,
            roi_mask=led_probability_mask_refined,
            config=config,
        )

    if "skeleton" in config["p4_fusion_branches"]:
        # Run skeleton branch
        lines_skeleton, skeleton_branch_results = _detect_lines_skeleton_branch(
            led_map=led_map,
            roi_mask=led_probability_mask_refined,
            threshold_value=config["p3_skeleton_threshold"],
            p2_roi_mask_morph_kernel_size=config["p3_skeleton_morph_kernel"],
            p2_roi_mask_morph_iterations=config["p3_skeleton_morph_iterations"],
            p3_edge_hough_threshold=config["p3_skeleton_hough_threshold"],
            p3_edge_min_line_length=config["p3_skeleton_min_line_length"],
            p3_edge_max_line_gap=config["p3_skeleton_max_line_gap"],
            p5_merge_angle_threshold=config["p5_merge_angle_threshold"],
            p5_merge_distance_threshold=config["p5_merge_distance_threshold"],
        )

    if "components" in config["p4_fusion_branches"]:
        # Run components branch
        lines_components, components_branch_results = _detect_lines_components_branch(
            led_map=led_map,
            roi_mask=led_probability_mask_refined,
            threshold_value=config["p3_skeleton_threshold"],
            p2_roi_mask_morph_kernel_size=config["p3_skeleton_morph_kernel"],
            p2_roi_mask_morph_iterations=config["p3_skeleton_morph_iterations"],
            min_component_area=config["p3_components_min_area"],
            min_aspect_ratio=config["p3_components_min_aspect_ratio"],
        )

    # Fuse results from all enabled branches
    lines_merged = _fuse_multiple_branches(
        edge_lines=lines_edge,
        skeleton_lines=lines_skeleton,
        components_lines=lines_components,
        angle_tolerance=config["p4_fusion_angle_tolerance"],
        distance_tolerance=config["p4_fusion_distance_tolerance"],
        min_overlap_ratio=config["p4_fusion_min_overlap"],
    )

    # Evaluate merged lines to get probabilities
    lines_merged_result = _evaluate_lines_by_led_probability(
        lines_merged,
        led_map,
        bright_thr=config["p6_filter_bright_threshold"],
        sampling_width=config["p6_filter_line_sampling_width"],
        num_samples=config["p6_filter_line_sampling_num_samples"],
        prob_high=config["p6_filter_prob_high_threshold"],
        prob_low=config["p6_filter_prob_low_threshold"],
    )
    lines_merged_probabilities = lines_merged_result["led_probabilities"]

    # Keep intermediate results for compatibility
    lines_detected = lines_merged  # For intermediate results compatibility
    lines_detected_probabilities = lines_merged_probabilities

    # Step 7: Final evaluation for filtering
    probability_result = _evaluate_lines_by_led_probability(
        lines_merged,
        led_map,
        bright_thr=config["p6_filter_bright_threshold"],
        sampling_width=config["p6_filter_line_sampling_width"],
        num_samples=config["p6_filter_line_sampling_num_samples"],
        prob_high=config["p6_filter_prob_high_threshold"],
        prob_low=config["p6_filter_prob_low_threshold"],
    )

    # Step 8: Filter lines by LED probability to remove unlikely LED candidates
    lines_final = lines_merged
    lines_final_probabilities = lines_merged_probabilities
    if config["p6_filter_by_led_probability"] and lines_merged is not None:
        lines_final, filtered_probs = _filter_lines_by_led_probability(
            lines_merged,
            np.array(probability_result["led_probabilities"]),
            p6_filter_min_led_probability=config["p6_filter_min_led_probability"],
            p6_filter_max_lines=config["p6_filter_max_lines"],
        )
        lines_final_probabilities = filtered_probs.tolist() if lines_final is not None else []

        # Update probability result to reflect filtered lines
        if lines_final is not None and len(filtered_probs) > 0:
            # Recompute statistics for filtered set using config thresholds
            high_prob = int((filtered_probs >= config["p6_filter_prob_high_threshold"]).sum())
            low_prob = int((filtered_probs <= config["p6_filter_prob_low_threshold"]).sum())
            medium_prob = len(filtered_probs) - high_prob - low_prob

            probability_result = LineLEDProbabilityResult(
                count=len(lines_final),
                led_probabilities=lines_final_probabilities,
                high_prob_lines=high_prob,
                low_prob_lines=low_prob,
                medium_prob_lines=medium_prob,
                mean_probability=float(filtered_probs.mean()),
            )

    # Create intermediate results object
    intermediate_results = PipelineIntermediateResults(
        original_image=image_bgr,
        led_map=led_map,
        led_probability_mask_unrefined=led_probability_mask_unrefined,
        led_probability_mask_refined=led_probability_mask_refined,
        edge_branch=edge_branch_results,
        skeleton_branch=skeleton_branch_results,
        components_branch=components_branch_results,
        lines_detected=lines_detected,
        lines_detected_probabilities=lines_detected_probabilities,
        lines_merged=lines_merged,
        lines_merged_probabilities=lines_merged_probabilities,
        lines_final=lines_final,
        lines_final_probabilities=lines_final_probabilities,
        probability_result=probability_result,
    )

    return lines_final, probability_result, intermediate_results


# ============================================================================
# Dual-Branch Line Detection: Edge Branch
# ============================================================================


def _detect_lines_edge_branch(
    image_bgr: np.ndarray,
    led_map: np.ndarray,
    roi_mask: np.ndarray,
    config: PipelineConfig,
) -> Tuple[Optional[np.ndarray], EdgeBranchResults]:
    """
    Edge-based line detection with paired-edge centerline extraction.

    This is the improved edge branch that solves the "two edges per strip" problem.

    Pipeline:
    1. Detect edges (Canny on LED map)
    2. Apply ROI mask
    3. Detect lines with Hough
    4. Find parallel pairs and compute centerlines (NEW)
    5. Merge remaining colinear lines

    Args:
        image_bgr: Original BGR image
        led_map: LED probability map
        roi_mask: ROI mask (refined LED probability mask)
        config: Pipeline configuration

    Returns:
        Tuple of (final_lines, edge_branch_results)
        - final_lines: Detected lines with paired edges merged into centerlines, or None
        - edge_branch_results: EdgeBranchResults containing all intermediate edge branch outputs
    """
    # Step 1: Detect edges (existing)
    edges_unmasked = _detect_edges(
        image_bgr,
        config["p3_edge_colorspace"],
        config["p3_edge_method"],
        p3_edge_canny_low=config["p3_edge_canny_low"],
        p3_edge_canny_high=config["p3_edge_canny_high"],
        p3_edge_blur_kernel=config["p3_edge_blur_kernel"],
        p3_edge_sobel_threshold=config["p3_edge_sobel_threshold"],
    )

    # Step 2: Apply LED probability mask to edges
    edges_masked = cv2.bitwise_and(edges_unmasked, edges_unmasked, mask=roi_mask)

    # Step 3: Detect lines with Hough (existing)
    lines_raw = _detect_lines_in_edges(
        edges_masked,
        config["p3_edge_line_method"],
        p3_edge_hough_threshold=config["p3_edge_hough_threshold"],
        p3_edge_min_line_length=config["p3_edge_min_line_length"],
        p3_edge_max_line_gap=config["p3_edge_max_line_gap"],
        p3_edge_preprocess_dilate=config["p3_edge_preprocess_dilate"],
        p3_edge_preprocess_erode=config["p3_edge_preprocess_erode"],
    )

    if lines_raw is None:
        # Return None for final lines, but still return edge detection results
        edge_results = EdgeBranchResults(
            edges_unmasked=edges_unmasked,
            edges_masked=edges_masked,
            lines_raw=None,
            lines_after_pairing=None,
            lines_after_merging=None,
        )
        return None, edge_results

    # Step 4: Find parallel pairs and compute centerlines (NEW)
    lines_with_centers = _find_parallel_pairs_and_centers(
        lines_raw,
        angle_threshold=config["p3_edge_pair_angle_threshold"],
        p3_edge_max_strip_width=config["p3_edge_max_strip_width"],
        min_overlap_ratio=config["p3_edge_min_pair_overlap"],
    )

    if lines_with_centers is None:
        # Return None for final lines, but still return edge detection results
        edge_results = EdgeBranchResults(
            edges_unmasked=edges_unmasked,
            edges_masked=edges_masked,
            lines_raw=lines_raw,
            lines_after_pairing=None,
            lines_after_merging=None,
        )
        return None, edge_results

    # Step 5: Merge remaining colinear lines (existing)
    lines_merged = _merge_collinear_lines(
        lines_with_centers,
        angle_threshold=config["p5_merge_angle_threshold"],
        distance_threshold=config["p5_merge_distance_threshold"],
    )

    # Construct edge branch results with all intermediate outputs
    edge_results = EdgeBranchResults(
        edges_unmasked=edges_unmasked,
        edges_masked=edges_masked,
        lines_raw=lines_raw,
        lines_after_pairing=lines_with_centers,
        lines_after_merging=lines_merged,
    )

    return lines_merged, edge_results


def _create_led_map(
    image_bgr: np.ndarray,
    p1_led_map_brightness_channel: Literal["gray", "hsv_v", "lab_l"],
    p1_led_map_use_clahe: bool,
    p1_led_map_bg_normalize: bool,
    p1_led_map_gamma: float,
    p1_led_map_use_color_weighting: bool,
    p1_led_map_color_mode: Literal["white", "hue"],
    p1_led_map_color_hue: float,
    p1_led_map_color_hue_tolerance: float,
    p1_led_map_max_white_saturation: int,
    p1_led_map_suppress_large_blobs: bool,
    p1_led_map_max_blob_area: int,
) -> np.ndarray:
    """
    Create an LED probability map where high values indicate likely LED locations.

    This function generates a brightness-based probability map for LED presence. It extracts
    a brightness channel from the input image and applies optimizations to enhance LED strips
    relative to the background. The output is a grayscale map where high pixel values (bright)
    correspond to likely LED locations and low values (dark) to non-LED areas.

    Args:
        image_bgr: BGR color image (OpenCV format)
        brightness_channel_type: Type of brightness channel to extract:
            - "gray": Standard grayscale conversion
            - "hsv_v": HSV Value channel (brightness component)
            - "lab_l": LAB Lightness channel (perceptual brightness)
        p1_led_map_use_clahe: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance
            local contrast and make LED strips more distinct
        use_background_norm: Apply background normalization to remove slow-varying illumination
            by subtracting estimated background (using large Gaussian blur)

    Returns:
        LED probability map (single-channel uint8 image) where:
        - High values (bright pixels) = likely LED locations
        - Low values (dark pixels) = non-LED areas

    Future optimization ideas:

    **Brightness Enhancement:**
    - Gamma correction with configurable gamma value for better LED/background separation
    - Adaptive histogram equalization with configurable clip limits and tile sizes
    - Local contrast enhancement using different window sizes (cv2.createCLAHE)
    - Contrast stretching / normalization to maximize dynamic range

    **Noise Reduction & Filtering:**
    - Gaussian blurring (cv2.GaussianBlur) to reduce noise before processing
    - Bilateral filtering (cv2.bilateralFilter) to preserve edges while smoothing
    - Morphological operations (opening/closing) to remove small artifacts

    **Advanced Illumination Modeling:**
    - Retinex algorithms for illumination-invariant representation
    - Multi-scale brightness analysis using image pyramids
    - Top-hat transform (cv2.morphologyEx MORPH_TOPHAT) for bright object extraction
    - Spectral unmixing to separate LED light from ambient light sources

    **Shape-Based LED Detection:**
    - Directional filtering: LEDs are linear, so brightness should be consistent along
      one direction but not perpendicular. Use oriented filters or line detectors.
    - Eliminate large bright surfaces (e.g., windows) that aren't LEDs using morphological
      area filtering or connected component analysis
    - Anisotropic diffusion to enhance linear structures while suppressing noise

    **Color-Based LED Filtering:**
    - Add configurable LED color parameter (e.g., white, warm white, cool white). Maybe just RGB value. Find something nice. 
    - Use color distance in RGB/HSV space to filter regions with colors far from LED color
    - Color constancy algorithms to normalize for varying ambient lighting
    - Saturation filtering (LEDs typically have low saturation if white)

    **Temporal Processing (for video):**
    - Temporal filtering to reduce noise and improve LED detection stability
    - Motion compensation for moving camera scenarios
    - Flicker detection for AC-powered LEDs
    - So this would mean adding the previous frame(s) as input to this function. 

    Tools/Libraries:
    - OpenCV: cv2.createCLAHE, cv2.GaussianBlur, cv2.bilateralFilter, cv2.morphologyEx
    - scikit-image: exposure.equalize_adapthist, restoration filters
    - scipy.ndimage: Gaussian filters, morphological operations
    """
    # Step 1: Extract raw brightness channel based on colorspace
    if p1_led_map_brightness_channel == "gray":
        channel = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    elif p1_led_map_brightness_channel == "hsv_v":
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        channel = hsv[:, :, 2]  # V (value/brightness) channel
    elif p1_led_map_brightness_channel == "lab_l":
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        channel = lab[:, :, 0]  # L (lightness) channel
    else:
        raise ValueError(f"Unknown p1_led_map_brightness_channel: {p1_led_map_brightness_channel}")

    # Step 2: Background normalization - removes slow-varying illumination
    # This helps separate bright LEDs from varying ambient lighting
    if p1_led_map_bg_normalize:
        channel_float = channel.astype(np.float32)
        # Large Gaussian blur estimates the background illumination
        blur = cv2.GaussianBlur(channel_float, (31, 31), 0)
        # Subtract background and normalize to 0-255 range
        normalized = channel_float - blur
        channel = np.zeros_like(normalized, dtype=np.uint8)
        cv2.normalize(normalized, channel, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Step 3: Gamma correction - boost bright pixels (LEDs) relative to medium/dark regions
    if p1_led_map_gamma > 0.0 and abs(p1_led_map_gamma - 1.0) > 1e-3:
        # Normalize to [0, 1] range
        channel_norm = channel.astype(np.float32) / 255.0
        # Apply gamma correction (values < 1.0 boost bright pixels)
        channel_gamma = np.power(channel_norm, p1_led_map_gamma)
        # Scale back to [0, 255]
        channel = (channel_gamma * 255.0).astype(np.uint8)

    # Step 4: CLAHE - enhances local contrast, making LEDs more distinct
    if p1_led_map_use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        channel = clahe.apply(channel)

    # Step 5: Color-based weighting (if enabled)
    if p1_led_map_use_color_weighting:
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0].astype(np.float32)  # Hue channel (0-180)
        S = hsv[:, :, 1].astype(np.float32)  # Saturation channel (0-255)

        # Step 6: Compute color weight map based on LED color mode
        if p1_led_map_color_mode == "white":
            # White LED mode: prefer low saturation (desaturated/white)
            # w_sat = 1 - clip(S / p1_led_map_max_white_saturation, 0, 1)
            # Low saturation → weight near 1.0, high saturation → weight near 0.0
            # Guard against divide-by-zero
            sat_den = float(max(p1_led_map_max_white_saturation, 1))
            w_color = 1.0 - np.clip(S / sat_den, 0.0, 1.0)
        else:  # p1_led_map_color_mode == "hue"
            # Hue LED mode: prefer specific hue range + high saturation
            # Compute circular hue distance (handle wraparound at 0/180)
            dh = np.abs(H - p1_led_map_color_hue)
            dh = np.minimum(dh, 180.0 - dh)
            # Hue weight: close to target hue → weight near 1.0
            # Guard against divide-by-zero
            tol = float(max(p1_led_map_color_hue_tolerance, 1e-3))
            w_h = np.clip(1.0 - dh / tol, 0.0, 1.0)
            # Saturation boost: high saturation → weight near 1.0
            # Use p1_led_map_max_white_saturation for consistency (with fallback)
            sat_den = float(max(p1_led_map_max_white_saturation, 1))
            w_sat = np.clip(S / sat_den, 0.0, 1.0)
            # Combined color weight
            w_color = w_h * w_sat

        # Step 7: Combine brightness and color weighting
        channel_float = channel.astype(np.float32)
        weighted = channel_float * w_color
        channel = np.clip(weighted, 0, 255).astype(np.uint8)

    # Step 9: Large blob suppression (remove windows, whiteboards)
    if p1_led_map_suppress_large_blobs:
        # Threshold to create binary mask
        _, binary = cv2.threshold(channel, 127, 255, cv2.THRESH_BINARY)
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        # Zero out large blobs
        for i in range(1, num_labels):  # Skip background label 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area > p1_led_map_max_blob_area:
                channel[labels == i] = 0

    return channel


def _compute_line_led_probability(
    led_map: np.ndarray,
    line: np.ndarray,
    bright_thr: int,
    sampling_width: int,
    num_samples: int,
) -> float:
    """
    Compute an LED strip probability score [0,1] for a detected line.

    This function analyzes the LED probability values along a detected line to determine how
    likely it is to be an LED strip. It requires an unmasked LED map so it can compare the
    line's LED probability against the surrounding background for contrast calculation.

    High probability lines (likely LED strips) should have:
    - High LED probability values along their length
    - Consistent LED probability (low std)
    - Higher LED probability than surrounding background

    Args:
        led_map: Unmasked LED probability map from _create_led_map(). High values indicate
            likely LED locations, low values indicate non-LED areas. Must NOT be masked so
            surroundings can be sampled for contrast measurement.
        line: Line coordinates [x1, y1, x2, y2]
        bright_thr: Brightness threshold for classifying pixels as "bright" (0-255)
        sampling_width: Width in pixels for perpendicular background sampling
        num_samples: Number of sample points along the line

    Returns:
        LED probability score in [0, 1] where 1 = perfect LED strip candidate
    """
    x1, y1, x2, y2 = line[0]

    line_vals, bg_vals = _sample_line_led_probability(
        led_map, x1, y1, x2, y2, width=sampling_width, num_samples=num_samples
    )

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
    q_bright = np.clip((mean_line - 150) / 80.0, 0.0, 1.0)  # 150-230 range
    q_ratio = np.clip((bright_ratio - 0.6) / 0.4, 0.0, 1.0)  # 60-100% bright pixels
    q_contrast = np.clip((contrast - 20) / 80.0, 0.0, 1.0)  # 20-100 contrast
    q_stable = 1.0 - np.clip((std_line - 10) / 40.0, 0.0, 1.0)  # Low std => stable

    # Weighted combination
    q = 0.35 * q_bright + 0.25 * q_ratio + 0.25 * q_contrast + 0.15 * q_stable
    return float(q)


def _detect_edges(
    unmasked_bgr_image: np.ndarray,
    p3_edge_colorspace: Literal["gray", "hsv"],
    method: str,
    p3_edge_canny_low: int,
    p3_edge_canny_high: int,
    p3_edge_blur_kernel: int,
    p3_edge_sobel_threshold: int,
) -> np.ndarray:
    """
    Detect edges in an unmasked BGR image optimized for LED strip detection.

    This function handles channel extraction, preprocessing, and edge detection. It accepts
    an unmasked BGR image to avoid detecting edges created by masking boundaries. The mask
    should be applied to the edges after detection, not before.

    Args:
        unmasked_bgr_image: Unmasked BGR color image (OpenCV format). Must NOT be masked
            to avoid detecting artificial edges created by mask boundaries.
        p3_edge_colorspace: Colorspace for edge detection ('gray' for grayscale, 'hsv' for HSV V channel)
        method: Edge detection method ('canny' or 'sobel')
        p3_edge_canny_low: Low threshold for Canny edge detection
        p3_edge_canny_high: High threshold for Canny edge detection
        p3_edge_blur_kernel: Size of Gaussian blur kernel (0 for no blur, must be odd if > 0)
        p3_edge_sobel_threshold: Threshold for Sobel edge detection

    Returns:
        Binary edge map (uint8)

    Future optimization ideas:
    - Histogram equalization for better contrast before edge detection
    - Adaptive histogram equalization (CLAHE)
    - Contrast stretching / normalization
    - Bilateral filtering to preserve edges while smoothing
    - Multi-scale edge detection (image pyramid)
    - Morphological gradient
    - Non-maximum suppression for thinning edges
    - Hysteresis thresholding for better edge connectivity
    """
    # Step 1: Extract channel for edge detection based on colorspace
    if p3_edge_colorspace == "gray":
        if unmasked_bgr_image.ndim == 3:
            channel = cv2.cvtColor(unmasked_bgr_image, cv2.COLOR_BGR2GRAY)
        else:
            channel = unmasked_bgr_image
    elif p3_edge_colorspace == "hsv":
        if unmasked_bgr_image.ndim == 3:
            hsv = cv2.cvtColor(unmasked_bgr_image, cv2.COLOR_BGR2HSV)
            channel = hsv[:, :, 2]  # V (value/brightness) channel
        else:
            channel = unmasked_bgr_image
    else:
        raise ValueError(f"Unknown p3_edge_colorspace: {p3_edge_colorspace}")

    # Step 2: Optional Gaussian blur before edge detection to reduce noise
    if p3_edge_blur_kernel > 0:
        # Ensure kernel is odd
        blur_k = p3_edge_blur_kernel
        if blur_k % 2 == 0:
            blur_k += 1
        channel = cv2.GaussianBlur(channel, (blur_k, blur_k), 0)

    if method == "canny":
        # Ensure uint8 for Canny
        if channel.dtype != np.uint8:
            normalized = np.zeros_like(channel, dtype=np.uint8)
            channel = cv2.normalize(
                channel, normalized, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )

        edges = cv2.Canny(channel, p3_edge_canny_low, p3_edge_canny_high)
        return edges
    elif method == "sobel":
        # Sobel edge detection
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        # Guard against div-by-zero
        max_val = magnitude.max()
        if max_val > 0:
            magnitude_normalized = magnitude / max_val * 255
        else:
            magnitude_normalized = np.zeros_like(magnitude, dtype=np.uint8)
        magnitude_uint8 = magnitude_normalized.astype(np.uint8)
        # Apply threshold to get binary edges
        _, edges = cv2.threshold(
            magnitude_uint8, p3_edge_sobel_threshold, 255, cv2.THRESH_BINARY
        )
        return edges
    else:
        raise ValueError(f"Unknown edge detection method: {method}")


def _evaluate_lines_by_led_probability(
    lines: Optional[np.ndarray],
    led_map: Optional[np.ndarray],
    bright_thr: int,
    sampling_width: int,
    num_samples: int,
    prob_high: float,
    prob_low: float,
) -> LineLEDProbabilityResult:
    """
    Evaluate detected lines by computing their LED strip probability scores.

    This function computes LED probability scores for detected lines by analyzing their
    LED probability profiles in the LED probability map. It requires an unmasked LED map to
    compare line LED probability against surrounding background for accurate contrast measurement.

    Args:
        lines: Detected lines array of shape (N, 1, 4) where each line is [[x1, y1, x2, y2]]
        led_map: Unmasked LED probability map from _create_led_map(). High values indicate
            likely LED locations, low values indicate non-LED areas. Must NOT be masked so
            surroundings can be sampled for contrast calculation. If None, returns zero
            probability scores.
        prob_high: Threshold for "high probability" LED candidates (>= this score)
        prob_low: Threshold for "low probability" lines (<= this score)

    Returns:
        LineLEDProbabilityResult with LED probability metrics (count, probabilities, distribution)
    """
    if lines is None or len(lines) == 0:
        return LineLEDProbabilityResult(
            count=0,
            led_probabilities=[],
            high_prob_lines=0,
            low_prob_lines=0,
            medium_prob_lines=0,
            mean_probability=0.0,
        )

    # Compute LED probabilities using the LED map
    if led_map is None:
        probabilities_array = np.zeros(
            len(lines)
        )  # Return zero probability scores if no LED map data
    else:
        # Compute LED probability scores for all lines
        probabilities = []
        for line in lines:
            prob = _compute_line_led_probability(
                led_map, line, bright_thr, sampling_width, num_samples
            )
            probabilities.append(prob)
        probabilities_array = np.array(probabilities)

    # Classify lines by LED probability
    high_prob_lines = int((probabilities_array >= prob_high).sum())
    low_prob_lines = int((probabilities_array <= prob_low).sum())
    medium_prob_lines = len(lines) - high_prob_lines - low_prob_lines

    # Overall mean probability
    mean_probability = float(probabilities_array.mean()) if len(probabilities_array) > 0 else 0.0

    return LineLEDProbabilityResult(
        count=len(lines),
        led_probabilities=probabilities_array.tolist(),
        high_prob_lines=high_prob_lines,
        low_prob_lines=low_prob_lines,
        medium_prob_lines=medium_prob_lines,
        mean_probability=mean_probability,
    )


def _filter_lines_by_led_probability(
    lines: Optional[np.ndarray],
    probabilities: np.ndarray,
    p6_filter_min_led_probability: float,
    p6_filter_max_lines: Optional[int],
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Filter detected lines based on pre-computed LED probability scores and limit to maximum number.

    This function filters out lines that are unlikely to be LED strips based on their
    LED probability scores (which measure how well each line matches LED characteristics).
    It can also limit the output to a maximum number of lines by keeping only the
    highest probability candidates.

    Args:
        lines: Detected lines array of shape (N, 1, 4) where each line is [[x1, y1, x2, y2]]
        probabilities: Pre-computed LED probability scores for each line (array of length N)
        p6_filter_min_led_probability: Minimum LED probability threshold [0,1]. Lines with scores
            below this threshold are filtered out.
        p6_filter_max_lines: Maximum number of lines to return. If specified and more lines pass
            the threshold, only the top max_lines highest probability lines are kept.
            If None, all lines above the threshold are returned.

    Returns:
        Tuple of (filtered_lines, filtered_probabilities):
        - filtered_lines: Filtered lines array of shape (M, 1, 4) where M <= N, or None if no lines pass
        - filtered_probabilities: Corresponding probability scores for filtered lines
    """
    if lines is None or len(lines) == 0:
        return None, np.array([])

    # Step 1: Filter by minimum LED probability threshold
    valid_mask = probabilities >= p6_filter_min_led_probability
    filtered_lines = lines[valid_mask]
    filtered_probs = probabilities[valid_mask]

    if len(filtered_lines) == 0:
        return None, np.array([])

    # Step 2: Limit to maximum number of lines if specified
    if p6_filter_max_lines is not None and len(filtered_lines) > p6_filter_max_lines:
        # Sort by probability (descending) and keep top p6_filter_max_lines
        top_indices = np.argsort(filtered_probs)[::-1][:p6_filter_max_lines]
        filtered_lines = filtered_lines[top_indices]
        filtered_probs = filtered_probs[top_indices]

    return filtered_lines, filtered_probs


def _refine_led_probability_mask(
    led_probability_mask: np.ndarray, kernel_size: int, iterations: int, p2_roi_mask_use_closing: bool
) -> np.ndarray:
    """Refine LED probability mask using morphological operations to connect gaps and smooth boundaries."""
    if kernel_size > 0 and iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        # Use closing (dilation + erosion) to connect gaps without excessive thickening
        if p2_roi_mask_use_closing:
            result = cv2.morphologyEx(
                led_probability_mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=iterations,
                borderType=cv2.BORDER_CONSTANT,
            )
        else:
            result = cv2.dilate(
                led_probability_mask, kernel, iterations=iterations, borderType=cv2.BORDER_CONSTANT
            )
        # Ensure output has same shape as input
        assert (
            result.shape == led_probability_mask.shape
        ), f"Shape mismatch: {result.shape} vs {led_probability_mask.shape}"
        return result
    return led_probability_mask


def _detect_lines_in_edges(
    masked_edge_image: np.ndarray,
    method: str,
    p3_edge_hough_threshold: int,
    p3_edge_min_line_length: int,
    p3_edge_max_line_gap: int,
    p3_edge_preprocess_dilate: int,
    p3_edge_preprocess_erode: int,
) -> Optional[np.ndarray]:
    """
    Detect lines in a masked edge image using the specified line detection method.

    This function receives a binary edge image that has been:
    1. Detected from an unmasked BGR image (to avoid artificial mask boundary edges)
    2. Masked with the LED probability mask to focus only on LED candidate regions

    The input should be a clean binary edge map with high values (255) on edges and low values (0)
    elsewhere, with no artificial edges from mask boundaries. This function optionally performs
    additional preprocessing on the edge image to optimize line detection quality.

    Args:
        masked_edge_image: Binary masked edge map (uint8) where:
            - High values (255) = edges in high LED probability regions (LED candidates)
            - Low values (0) = non-edge pixels or regions outside LED candidates
            This image should NOT have artificial edges from mask boundaries.
        method: Line detection method:
            - 'hough': Probabilistic Hough Transform (cv2.HoughLinesP) - fast, good for straight lines
            - 'lsd': Line Segment Detector - more accurate but slower
        p3_edge_hough_threshold: Hough transform accumulator threshold. Higher values require more
            evidence for a line (fewer false positives, may miss weak lines)
        p3_edge_min_line_length: Minimum line length in pixels. Shorter segments are rejected.
        p3_edge_max_line_gap: Maximum gap between line segments to merge them. Useful for connecting
            broken LED lines due to gaps or occlusions.
        p3_edge_preprocess_dilate: Dilation kernel size for connecting nearby edges before line detection.
            Set to 0 to disable. Useful for connecting broken edges from discontinuous LEDs.
        p3_edge_preprocess_erode: Erosion kernel size for thinning edges before line detection.
            Set to 0 to disable. Useful for removing noise and thinning thick edges.

    Returns:
        Array of shape (N, 1, 4) where each line is [[x1, y1, x2, y2]], or None if no lines detected

    Future optimization ideas:
    - Non-maximum suppression to thin edges and improve line detection accuracy
    - Hysteresis thresholding with dual thresholds for better edge connectivity
    - Morphological gradient for additional edge enhancement
    - Distance transform to prioritize strong edge centers
    - Edge orientation filtering to focus on horizontal/vertical lines if LEDs have known orientation
    """
    # Step 1: Optional preprocessing to optimize edge image for line detection
    processed_edges = masked_edge_image.copy()

    # Dilation: Connect nearby edges (useful for broken/discontinuous LED lines)
    if p3_edge_preprocess_dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (p3_edge_preprocess_dilate, p3_edge_preprocess_dilate))
        processed_edges = cv2.dilate(processed_edges, kernel, iterations=1)

    # Erosion: Thin edges and remove noise (useful for cleaning up thick edges)
    if p3_edge_preprocess_erode > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (p3_edge_preprocess_erode, p3_edge_preprocess_erode))
        processed_edges = cv2.erode(processed_edges, kernel, iterations=1)

    # Step 2: Line detection using specified method
    if method == "hough":
        return cv2.HoughLinesP(
            processed_edges,
            1,
            np.pi / 180,
            threshold=p3_edge_hough_threshold,
            minLineLength=p3_edge_min_line_length,
            maxLineGap=p3_edge_max_line_gap,
        )
    elif method == "lsd":
        # Line Segment Detector
        lsd = cv2.createLineSegmentDetector(0)
        lines, _, _, _ = lsd.detect(processed_edges)
        if lines is not None:
            # Convert to HoughLinesP format for consistency
            lines = lines.reshape(-1, 1, 4).astype(int)
        return lines
    else:
        raise ValueError(f"Unknown line detection method: {method}")


def _point_to_line_distance(
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


def _create_led_probability_mask(
    led_map: np.ndarray,
    threshold: int,
    p2_roi_mask_use_adaptive: bool,
    p2_roi_mask_top_ratio: float,
) -> np.ndarray:
    """
    Create a binary LED probability mask identifying regions likely to contain LEDs.

    This function receives an LED probability map (created by _create_led_map()) where high
    values indicate likely LED locations. It applies thresholding and optional ROI filtering
    to create a binary mask of candidate regions for further processing.

    Args:
        led_map: Unmasked LED probability map from _create_led_map(). High values indicate
            likely LED locations, low values indicate non-LED areas. Must NOT be masked
            so the full map is available for proper thresholding.
        threshold: LED probability threshold for binary thresholding (0-255). Pixels above
            this value are considered LED candidates.
        p2_roi_mask_use_adaptive: Use adaptive thresholding instead of global thresholding. Adaptive
            thresholding can handle varying illumination conditions better.
        p2_roi_mask_top_ratio: Ratio of image height to use as Region of Interest (0.0-1.0).
            For ceiling-mounted LEDs, set to 0.5-0.6 to focus on top portion of image.
            0.0 = use full image.

    Returns:
        Binary LED probability mask (uint8) where:
        - 255 (white) = high LED probability regions (LED candidates)
        - 0 (black) = low LED probability regions or areas outside ROI

    Future optimization ideas:
    - Hysteresis thresholding (dual threshold like Canny) for better region detection
    - Connected component analysis to filter by size/aspect ratio
    - Distance transform to find strong LED centers
    - Watershed segmentation for separating overlapping LED regions
    """

    # Step 1: Thresholding (adaptive or global)
    if p2_roi_mask_use_adaptive:
        mask = cv2.adaptiveThreshold(
            led_map,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,  # block size
            -5,  # C constant
        )
    else:
        _, mask = cv2.threshold(led_map, threshold, 255, cv2.THRESH_BINARY)

    # Step 2: Apply ROI mask (focus on top portion of image for ceiling LEDs)
    if p2_roi_mask_top_ratio > 0:
        roi_mask = np.zeros_like(mask)
        height = mask.shape[0]
        roi_height = int(height * p2_roi_mask_top_ratio)
        roi_mask[:roi_height, :] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

    return mask


def _sample_line_led_probability(
    led_map: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    num_samples: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Sample LED probability values along a line and in perpendicular strips for contrast comparison.

    This function samples values from an LED probability map (created by _create_led_map()) along
    a detected line and in perpendicular strips on both sides. The line samples represent the LED
    candidate's probability values, while the perpendicular samples represent the surrounding background.
    This allows calculation of contrast between the line and its background.

    Args:
        led_map: Unmasked LED probability map from _create_led_map(). High values indicate likely
            LED locations, low values indicate non-LED areas. Must NOT be masked so we can sample
            the surrounding background for contrast measurement.
        x1, y1: Start coordinates of the line
        x2, y2: End coordinates of the line
        width: Width of perpendicular sampling strip (pixels on each side). Larger values sample
            more background but may include other bright regions.
        num_samples: Number of sample points along the line. More samples give better statistics
            but are slower to compute.

    Returns:
        Tuple of (line_values, background_values):
        - line_values: Array of LED probability values sampled along the line, or None if invalid
        - background_values: Array of values sampled perpendicular to the line (background), or None
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

    H, W = led_map.shape
    for x, y in zip(xs, ys):
        x_c = int(round(x))
        y_c = int(round(y))
        if 0 <= x_c < W and 0 <= y_c < H:
            line_vals.append(led_map[y_c, x_c])

            # Sample both sides perpendicular to line for background comparison
            for k in range(1, width + 1):
                xp1 = int(round(x + nx * k))
                yp1 = int(round(y + ny * k))
                xp2 = int(round(x - nx * k))
                yp2 = int(round(y - ny * k))
                if 0 <= xp1 < W and 0 <= yp1 < H:
                    bg_vals.append(led_map[yp1, xp1])
                if 0 <= xp2 < W and 0 <= yp2 < H:
                    bg_vals.append(led_map[yp2, xp2])

    if not line_vals:
        return None, None
    return np.array(line_vals), np.array(bg_vals) if bg_vals else None


# ============================================================================
# Dual-Branch Line Detection: Helper Functions
# ============================================================================


def _compute_line_angle(line: np.ndarray) -> float:
    """
    Compute the angle of a line in degrees [0, 180).

    Args:
        line: Line as [[x1, y1, x2, y2]]

    Returns:
        Angle in degrees, normalized to [0, 180) range
    """
    x1, y1, x2, y2 = line[0]
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad)

    # Normalize to [0, 180) - we don't care about direction
    angle_deg = angle_deg % 180
    return float(angle_deg)


def _compute_perpendicular_distance(line1: np.ndarray, line2: np.ndarray) -> float:
    """
    Compute perpendicular distance between two lines.

    Uses the midpoint of line2 and computes its distance to line1.

    Args:
        line1: First line [[x1, y1, x2, y2]]
        line2: Second line [[x1, y1, x2, y2]]

    Returns:
        Perpendicular distance in pixels
    """
    x1, y1, x2, y2 = line1[0]
    mx, my = (line2[0][0] + line2[0][2]) / 2, (line2[0][1] + line2[0][3]) / 2

    # Line direction vector
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)

    if length < 1e-6:
        # Degenerate line, just compute distance to point
        return float(np.sqrt((mx - x1) ** 2 + (my - y1) ** 2))

    # Normalize direction
    dx /= length
    dy /= length

    # Vector from line1 start to midpoint of line2
    vx = mx - x1
    vy = my - y1

    # Project onto line direction
    t = vx * dx + vy * dy

    # Closest point on line1 (infinite line through line1's segment)
    px = x1 + t * dx
    py = y1 + t * dy

    # Distance from midpoint to closest point
    dist = np.sqrt((mx - px) ** 2 + (my - py) ** 2)
    return float(dist)


def _compute_overlap_ratio(line1: np.ndarray, line2: np.ndarray) -> float:
    """
    Compute overlap ratio between two lines along their direction.

    Projects both lines onto the direction of line1 and computes
    the ratio of overlapping length to total span.

    Args:
        line1: First line [[x1, y1, x2, y2]]
        line2: Second line [[x1, y1, x2, y2]]

    Returns:
        Overlap ratio in [0, 1] where 1 = complete overlap
    """
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    # Direction of line1
    dx = x2 - x1
    dy = y2 - y1
    length1 = np.sqrt(dx * dx + dy * dy)

    if length1 < 1e-6:
        return 0.0

    # Normalize direction
    dx /= length1
    dy /= length1

    # Project all four endpoints onto line1's direction
    t1 = 0.0  # line1 start
    t2 = length1  # line1 end
    t3 = (x3 - x1) * dx + (y3 - y1) * dy  # line2 start
    t4 = (x4 - x1) * dx + (y4 - y1) * dy  # line2 end

    # Sort line2 projections
    if t3 > t4:
        t3, t4 = t4, t3

    # Compute overlap interval
    overlap_start = max(t1, t3)
    overlap_end = min(t2, t4)
    overlap_length = max(0.0, overlap_end - overlap_start)

    # Total span
    total_start = min(t1, t3)
    total_end = max(t2, t4)
    total_length = total_end - total_start

    if total_length < 1e-6:
        return 0.0

    return float(overlap_length / total_length)


def _find_parallel_pairs_and_centers(
    lines: np.ndarray,
    angle_threshold: float = 3.0,
    p3_edge_max_strip_width: float = 20.0,
    min_overlap_ratio: float = 0.7,
) -> Optional[np.ndarray]:
    """
    Find pairs of parallel lines (edge pairs) and replace with centerlines.

    This solves the "two edges per strip" problem by detecting parallel line pairs
    that likely represent the left and right edges of the same LED strip, then
    computing a single centerline for each pair.

    Algorithm:
    1. For each line, find all parallel candidates (similar angle, close distance, good overlap)
    2. Greedily pair lines, prioritizing closest pairs
    3. For each pair, compute centerline (average offset, combined span)
    4. Keep unpaired lines as-is

    Args:
        lines: Detected lines from Hough [[x1,y1,x2,y2], ...]
        angle_threshold: Max angle difference for parallelism (degrees)
        p3_edge_max_strip_width: Max perpendicular distance for pairing (pixels)
        min_overlap_ratio: Min overlap along direction (0-1)

    Returns:
        New line set with paired edges replaced by centerlines
    """
    if lines is None or len(lines) == 0:
        return lines

    n = len(lines)
    paired = set()
    result = []

    # Compute properties for all lines
    angles = np.array([_compute_line_angle(line) for line in lines])

    # Try to pair each line with its best candidate
    for i in range(n):
        if i in paired:
            continue

        line1 = lines[i]
        angle1 = angles[i]
        best_j = None
        best_dist = float('inf')

        # Find best pairing candidate
        for j in range(i + 1, n):
            if j in paired:
                continue

            line2 = lines[j]
            angle2 = angles[j]

            # Check angle similarity
            angle_diff = abs(angle1 - angle2)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            if angle_diff > angle_threshold:
                continue

            # Check perpendicular distance
            dist = _compute_perpendicular_distance(line1, line2)
            if dist > p3_edge_max_strip_width:
                continue

            # Check overlap
            overlap = _compute_overlap_ratio(line1, line2)
            if overlap < min_overlap_ratio:
                continue

            # This is a valid candidate, track the closest one
            if dist < best_dist:
                best_dist = dist
                best_j = j

        # If we found a pair, create centerline
        if best_j is not None:
            line2 = lines[best_j]
            centerline = _create_centerline(line1, line2)
            result.append(centerline)
            paired.add(i)
            paired.add(best_j)
        else:
            # No pair found, keep original line
            result.append(line1)
            paired.add(i)

    if len(result) == 0:
        return None

    return np.array(result).reshape(-1, 1, 4)


def _create_centerline(line1: np.ndarray, line2: np.ndarray) -> np.ndarray:
    """
    Create a centerline between two parallel lines.

    Strategy:
    1. Compute average direction from both lines
    2. Project all 4 endpoints onto this direction
    3. Take min/max projections to get the full span
    4. Compute center offset (average of both line offsets)
    5. Construct centerline at center offset spanning min to max

    Args:
        line1: First line [[x1, y1, x2, y2]]
        line2: Second line [[x1, y1, x2, y2]]

    Returns:
        Centerline [[x1, y1, x2, y2]]
    """
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    # Compute direction vectors
    dx1, dy1 = x2 - x1, y2 - y1
    length1 = np.sqrt(dx1 * dx1 + dy1 * dy1)

    dx2, dy2 = x4 - x3, y4 - y3
    length2 = np.sqrt(dx2 * dx2 + dy2 * dy2)

    if length1 < 1e-6 or length2 < 1e-6:
        # Degenerate line, just average endpoints
        cx1 = int((x1 + x3) / 2)
        cy1 = int((y1 + y3) / 2)
        cx2 = int((x2 + x4) / 2)
        cy2 = int((y2 + y4) / 2)
        return np.array([[cx1, cy1, cx2, cy2]])

    # Normalize directions
    dx1 /= length1
    dy1 /= length1
    dx2 /= length2
    dy2 /= length2

    # Average direction (ensure pointing same way)
    if dx1 * dx2 + dy1 * dy2 < 0:
        dx2, dy2 = -dx2, -dy2

    dx_avg = (dx1 + dx2) / 2
    dy_avg = (dy1 + dy2) / 2
    length_avg = np.sqrt(dx_avg * dx_avg + dy_avg * dy_avg)
    dx_avg /= length_avg
    dy_avg /= length_avg

    # Average midpoint (reference point for projections)
    mid1_x, mid1_y = (x1 + x2) / 2, (y1 + y2) / 2
    mid2_x, mid2_y = (x3 + x4) / 2, (y3 + y4) / 2
    ref_x = (mid1_x + mid2_x) / 2
    ref_y = (mid1_y + mid2_y) / 2

    # Project all endpoints onto average direction
    endpoints = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    projections = []
    for px, py in endpoints:
        t = (px - ref_x) * dx_avg + (py - ref_y) * dy_avg
        projections.append(t)

    # Get span
    t_min = min(projections)
    t_max = max(projections)

    # Construct centerline endpoints
    cx1 = ref_x + t_min * dx_avg
    cy1 = ref_y + t_min * dy_avg
    cx2 = ref_x + t_max * dx_avg
    cy2 = ref_y + t_max * dy_avg

    return np.array([[int(cx1), int(cy1), int(cx2), int(cy2)]])


# ============================================================================
# Dual-Branch Line Detection: Region Branch
# ============================================================================


def _fit_line_pca(points: np.ndarray) -> Optional[np.ndarray]:
    """
    Fit a line to 2D points using PCA.

    Args:
        points: (N, 2) array of [x, y] coordinates

    Returns:
        Line as [[x1, y1, x2, y2]] or None if fit fails
    """
    if len(points) < 2:
        return None

    # 1. Compute mean and center points
    mean = points.mean(axis=0)
    centered = points - mean

    # 2. Compute covariance matrix
    cov = np.cov(centered.T)

    # 3. Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eig(cov)

    # 4. Principal direction = eigenvector with largest eigenvalue
    idx = np.argmax(eigvals)
    direction = eigvecs[:, idx]  # (dx, dy)

    # 5. Project all points onto principal axis to find extent
    projections = centered.dot(direction)
    t_min = projections.min()
    t_max = projections.max()

    # 6. Compute endpoints
    p1 = mean + t_min * direction
    p2 = mean + t_max * direction

    return np.array([[int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])]])


def _detect_lines_components_pca(
    mask: np.ndarray,
    min_component_area: int,
    min_aspect_ratio: float,
) -> Optional[np.ndarray]:
    """
    Find connected components and fit line per component via PCA.

    Args:
        mask: Binary mask of LED regions
        min_component_area: Minimum component area in pixels
        min_aspect_ratio: Minimum aspect ratio for elongated regions

    Returns:
        Line array [[x1,y1,x2,y2], ...] or None
    """
    # 1. Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    lines = []

    # 2. For each component (skip background label 0)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        # Filter by area
        if area < min_component_area:
            continue

        # Get bounding box for aspect ratio check
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        aspect = max(w, h) / max(min(w, h), 1)

        # Filter by aspect ratio (elongated regions only)
        if aspect < min_aspect_ratio:
            continue

        # 3. Extract all pixels of this component
        ys, xs = np.where(labels == label_id)
        pts = np.column_stack((xs, ys)).astype(np.float32)  # (N, 2)

        # 4. Fit line via PCA
        line = _fit_line_pca(pts)
        if line is not None:
            lines.append(line)

    if len(lines) == 0:
        return None

    return np.array(lines).reshape(-1, 1, 4)


def _detect_lines_skeleton_hough(
    mask: np.ndarray,
    p3_edge_hough_threshold: int,
    p3_edge_min_line_length: int,
    p3_edge_max_line_gap: int,
    p5_merge_angle_threshold: float,
    p5_merge_distance_threshold: float,
) -> Optional[np.ndarray]:
    """
    Skeletonize mask and detect lines with Hough.

    Args:
        mask: Binary mask of LED regions
        p3_edge_hough_threshold: Hough transform threshold
        p3_edge_min_line_length: Minimum line length
        p3_edge_max_line_gap: Maximum line gap
        p5_merge_angle_threshold: Angle threshold for merging
        p5_merge_distance_threshold: Distance threshold for merging

    Returns:
        Line array [[x1,y1,x2,y2], ...] or None
    """
    # 1. Skeletonize (thin to 1-pixel centerlines)
    # Try to use cv2.ximgproc.thinning if available, otherwise use morphological thinning
    try:
        skeleton = cv2.ximgproc.thinning(mask)
    except AttributeError:
        # Fallback: morphological thinning (less optimal but works)
        skeleton = mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(skeleton, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(skeleton, temp)
            skeleton = eroded
            if cv2.countNonZero(temp) == 0:
                break

    # 2. Hough on skeleton
    lines = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi/180,
        threshold=p3_edge_hough_threshold,
        minLineLength=p3_edge_min_line_length,
        maxLineGap=p3_edge_max_line_gap,
    )

    if lines is None:
        return None

    # 3. Light merge (skeleton fragments)
    lines_merged = _merge_collinear_lines(
        lines,
        angle_threshold=p5_merge_angle_threshold,
        distance_threshold=p5_merge_distance_threshold,
    )

    return lines_merged


def _detect_lines_skeleton_branch(
    led_map: np.ndarray,
    roi_mask: np.ndarray,
    threshold_value: int,
    p2_roi_mask_morph_kernel_size: int,
    p2_roi_mask_morph_iterations: int,
    p3_edge_hough_threshold: int,
    p3_edge_min_line_length: int,
    p3_edge_max_line_gap: int,
    p5_merge_angle_threshold: float,
    p5_merge_distance_threshold: float,
) -> Tuple[Optional[np.ndarray], RegionBranchResults]:
    """
    Skeleton-based line detection from LED probability map.

    Pipeline:
    1. Threshold LED map to binary mask
    2. Morphological cleanup
    3. Skeletonize regions to 1-pixel centerlines
    4. Run Hough on skeleton
    5. Merge colinear skeleton fragments

    Args:
        led_map: LED probability map
        roi_mask: ROI mask to limit detection region
        threshold_value: Threshold for LED map (0-255)
        p2_roi_mask_morph_kernel_size: Morphology kernel size
        p2_roi_mask_morph_iterations: Morphology iterations
        p3_edge_hough_threshold: Hough threshold
        p3_edge_min_line_length: Min line length
        p3_edge_max_line_gap: Max line gap
        p5_merge_angle_threshold: Angle threshold for merging
        p5_merge_distance_threshold: Distance threshold for merging

    Returns:
        Tuple of (lines, results) where:
        - lines: Line array [[x1,y1,x2,y2], ...] or None
        - results: RegionBranchResults containing intermediate visualizations
    """
    # 1. Threshold LED map to binary mask
    _, mask = cv2.threshold(led_map, threshold_value, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, mask, mask=roi_mask)

    # 2. Morphological cleanup
    if p2_roi_mask_morph_kernel_size > 0 and p2_roi_mask_morph_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (p2_roi_mask_morph_kernel_size, p2_roi_mask_morph_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=p2_roi_mask_morph_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. Skeletonize mask
    try:
        skeleton = cv2.ximgproc.thinning(mask)
    except AttributeError:
        # Fallback: morphological thinning
        skeleton = mask.copy()
        kernel_thin = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(skeleton, kernel_thin)
            temp = cv2.dilate(eroded, kernel_thin)
            temp = cv2.subtract(skeleton, temp)
            skeleton = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break

    # 4. Detect lines using skeleton Hough
    lines = _detect_lines_skeleton_hough(
        mask,
        p3_edge_hough_threshold=p3_edge_hough_threshold,
        p3_edge_min_line_length=p3_edge_min_line_length,
        p3_edge_max_line_gap=p3_edge_max_line_gap,
        p5_merge_angle_threshold=p5_merge_angle_threshold,
        p5_merge_distance_threshold=p5_merge_distance_threshold,
    )

    # Create results structure
    results = RegionBranchResults(
        region_mask=mask,
        skeleton=skeleton,
        lines_detected=lines,
    )

    return lines, results


def _detect_lines_components_branch(
    led_map: np.ndarray,
    roi_mask: np.ndarray,
    threshold_value: int,
    p2_roi_mask_morph_kernel_size: int,
    p2_roi_mask_morph_iterations: int,
    min_component_area: int,
    min_aspect_ratio: float,
) -> Tuple[Optional[np.ndarray], RegionBranchResults]:
    """
    Component-based line detection from LED probability map.

    Pipeline:
    1. Threshold LED map to binary mask
    2. Morphological cleanup
    3. Find connected components
    4. Filter by area and aspect ratio
    5. Fit line per component via PCA

    Args:
        led_map: LED probability map
        roi_mask: ROI mask to limit detection region
        threshold_value: Threshold for LED map (0-255)
        p2_roi_mask_morph_kernel_size: Morphology kernel size
        p2_roi_mask_morph_iterations: Morphology iterations
        min_component_area: Min component area (pixels)
        min_aspect_ratio: Min aspect ratio for elongated regions

    Returns:
        Tuple of (lines, results) where:
        - lines: Line array [[x1,y1,x2,y2], ...] or None
        - results: RegionBranchResults containing intermediate visualizations
    """
    # 1. Threshold LED map to binary mask
    _, mask = cv2.threshold(led_map, threshold_value, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, mask, mask=roi_mask)

    # 2. Morphological cleanup
    if p2_roi_mask_morph_kernel_size > 0 and p2_roi_mask_morph_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (p2_roi_mask_morph_kernel_size, p2_roi_mask_morph_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=p2_roi_mask_morph_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. Detect lines using component PCA
    lines = _detect_lines_components_pca(
        mask,
        min_component_area=min_component_area,
        min_aspect_ratio=min_aspect_ratio,
    )

    # Create results structure (no skeleton for components mode)
    results = RegionBranchResults(
        region_mask=mask,
        skeleton=None,
        lines_detected=lines,
    )

    return lines, results


# ============================================================================
# Dual-Branch Line Detection: Fusion
# ============================================================================


def _are_lines_similar(
    line1: np.ndarray,
    line2: np.ndarray,
    angle_tolerance: float,
    distance_tolerance: float,
    min_overlap_ratio: float,
) -> bool:
    """
    Check if two lines are similar (parallel, close, overlapping).

    Used during fusion to detect redundant lines from different branches.

    Args:
        line1: First line [[x1, y1, x2, y2]]
        line2: Second line [[x1, y1, x2, y2]]
        angle_tolerance: Max angle difference (degrees)
        distance_tolerance: Max perpendicular distance (pixels)
        min_overlap_ratio: Min overlap ratio (0-1)

    Returns:
        True if lines represent the same strip
    """
    # 1. Compute angles
    angle1 = _compute_line_angle(line1)
    angle2 = _compute_line_angle(line2)
    angle_diff = abs(angle1 - angle2)

    # Handle wraparound (179° vs 1° should be 2° diff)
    if angle_diff > 90:
        angle_diff = 180 - angle_diff

    if angle_diff > angle_tolerance:
        return False

    # 2. Compute perpendicular distance
    dist = _compute_perpendicular_distance(line1, line2)
    if dist > distance_tolerance:
        return False

    # 3. Compute overlap ratio along direction
    overlap = _compute_overlap_ratio(line1, line2)
    if overlap < min_overlap_ratio:
        return False

    return True


def _fuse_multiple_branches(
    edge_lines: Optional[np.ndarray],
    skeleton_lines: Optional[np.ndarray],
    components_lines: Optional[np.ndarray],
    angle_tolerance: float = 3.0,
    distance_tolerance: float = 10.0,
    min_overlap_ratio: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Fuse lines from edge, skeleton, and components branches.

    Strategy:
    - Region-based lines (skeleton, components) are primary (they represent true centerlines)
    - Priority order: components > skeleton > edge
    - Lines are added only if they provide new information (not similar to already added lines)

    Args:
        edge_lines: Lines from edge branch (Hough on edges)
        skeleton_lines: Lines from skeleton branch (skeleton + Hough)
        components_lines: Lines from components branch (PCA on components)
        angle_tolerance: Max angle diff to consider lines similar (degrees)
        distance_tolerance: Max perpendicular distance to consider overlap (pixels)
        min_overlap_ratio: Min overlap along direction to consider redundant

    Returns:
        Fused line set
    """
    # Collect all non-None line sets with their priority (higher = more trusted)
    branch_lines = []
    if components_lines is not None:
        branch_lines.append((3, components_lines))  # Highest priority
    if skeleton_lines is not None:
        branch_lines.append((2, skeleton_lines))
    if edge_lines is not None:
        branch_lines.append((1, edge_lines))  # Lowest priority

    # Handle empty case
    if len(branch_lines) == 0:
        return None

    # Sort by priority (highest first)
    branch_lines.sort(key=lambda x: x[0], reverse=True)

    # Start with highest priority lines
    fused = []
    for line in branch_lines[0][1]:
        fused.append(line)

    # Add lines from lower priority branches if they're not redundant
    for priority, lines in branch_lines[1:]:
        for line in lines:
            is_redundant = False

            # Check if this line is similar to any already fused line
            for fused_line in fused:
                if _are_lines_similar(
                    line,
                    fused_line,
                    angle_tolerance=angle_tolerance,
                    distance_tolerance=distance_tolerance,
                    min_overlap_ratio=min_overlap_ratio,
                ):
                    is_redundant = True
                    break

            # Only add if it's new information
            if not is_redundant:
                fused.append(line)

    if len(fused) == 0:
        return None

    return np.array(fused).reshape(-1, 1, 4)


def _fuse_edge_and_region_lines(
    edge_lines: Optional[np.ndarray],
    region_lines: Optional[np.ndarray],
    angle_tolerance: float = 3.0,
    distance_tolerance: float = 10.0,
    min_overlap_ratio: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Fuse lines from edge and region branches.

    DEPRECATED: Use _fuse_multiple_branches instead.
    This function is kept for backward compatibility.

    Strategy:
    - Region lines are primary (they represent true centerlines)
    - Edge lines are added only if they provide new information

    Args:
        edge_lines: Lines from edge branch (Hough on edges)
        region_lines: Lines from region branch (centerlines)
        angle_tolerance: Max angle diff to consider lines similar (degrees)
        distance_tolerance: Max perpendicular distance to consider overlap (pixels)
        min_overlap_ratio: Min overlap along direction to consider redundant

    Returns:
        Fused line set
    """
    return _fuse_multiple_branches(
        edge_lines=edge_lines,
        skeleton_lines=region_lines,
        components_lines=None,
        angle_tolerance=angle_tolerance,
        distance_tolerance=distance_tolerance,
        min_overlap_ratio=min_overlap_ratio,
    )


def _merge_collinear_lines(
    lines: Optional[np.ndarray],
    angle_threshold: float,
    distance_threshold: float,
) -> Optional[np.ndarray]:
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
            dist1 = _point_to_line_distance(x3, y3, x1, y1, x2, y2)
            dist2 = _point_to_line_distance(x4, y4, x1, y1, x2, y2)

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
