#!/usr/bin/env python3
"""
Rename pipeline configuration parameters to have descriptive prefixes.
Format: p<phase>_<section>_<parameter_name>

Phase IDs:
- p1: LED Map Creation
- p2: ROI Masking
- p3: Edge/Skeleton/Components Detection
- p4: Branch Fusion
- p5: Line Merging
- p6: Line Evaluation and Filtering
"""

import re
import os

# Define the renaming mappings
# Format: old_name -> new_name
RENAMES = {
    # Phase 1: LED Map Creation (already partially done)
    # p1_led_map_brightness_channel - already done
    # p1_led_map_bg_normalize - already done
    "gamma": "p1_led_map_gamma",
    "use_clahe": "p1_led_map_use_clahe",
    "use_color_weighting": "p1_led_map_use_color_weighting",
    "led_color_mode": "p1_led_map_color_mode",
    "led_color_hue": "p1_led_map_color_hue",
    "led_color_hue_tolerance": "p1_led_map_color_hue_tolerance",
    "max_white_saturation": "p1_led_map_max_white_saturation",
    "suppress_large_blobs": "p1_led_map_suppress_large_blobs",
    "max_blob_area": "p1_led_map_max_blob_area",

    # Phase 2: ROI Masking
    "led_probability_threshold": "p2_roi_mask_threshold",
    "use_adaptive": "p2_roi_mask_use_adaptive",
    "roi_top_ratio": "p2_roi_mask_top_ratio",
    "morph_kernel_size": "p2_roi_mask_morph_kernel_size",
    "morph_iterations": "p2_roi_mask_morph_iterations",
    "use_closing": "p2_roi_mask_use_closing",

    # Phase 3: Edge Detection (for edge branch)
    "colorspace": "p3_edge_colorspace",
    "edge_method": "p3_edge_method",
    "canny_low": "p3_edge_canny_low",
    "canny_high": "p3_edge_canny_high",
    "blur_kernel": "p3_edge_blur_kernel",
    "sobel_threshold": "p3_edge_sobel_threshold",
    "line_method": "p3_edge_line_method",
    "hough_threshold": "p3_edge_hough_threshold",
    "min_line_length": "p3_edge_min_line_length",
    "max_line_gap": "p3_edge_max_line_gap",
    "preprocess_dilate": "p3_edge_preprocess_dilate",
    "preprocess_erode": "p3_edge_preprocess_erode",
    "pair_angle_threshold": "p3_edge_pair_angle_threshold",
    "max_strip_width": "p3_edge_max_strip_width",
    "min_pair_overlap": "p3_edge_min_pair_overlap",

    # Phase 3: Skeleton Branch
    "region_threshold": "p3_skeleton_threshold",
    "region_morph_kernel": "p3_skeleton_morph_kernel",
    "region_morph_iterations": "p3_skeleton_morph_iterations",
    "region_hough_threshold": "p3_skeleton_hough_threshold",
    "region_min_line_length": "p3_skeleton_min_line_length",
    "region_max_line_gap": "p3_skeleton_max_line_gap",

    # Phase 3: Components Branch (shares some params with skeleton)
    "region_min_area": "p3_components_min_area",
    "region_min_aspect_ratio": "p3_components_min_aspect_ratio",

    # Phase 4: Branch Selection and Fusion
    "branches": "p4_fusion_branches",
    "fusion_angle_tolerance": "p4_fusion_angle_tolerance",
    "fusion_distance_tolerance": "p4_fusion_distance_tolerance",
    "fusion_min_overlap": "p4_fusion_min_overlap",

    # Phase 5: Line Merging
    "merge_lines": "p5_merge_enabled",
    "merge_angle_threshold": "p5_merge_angle_threshold",
    "merge_distance_threshold": "p5_merge_distance_threshold",

    # Phase 6: Line Evaluation and Filtering
    "filter_by_led_probability": "p6_filter_by_led_probability",
    "min_led_probability": "p6_filter_min_led_probability",
    "max_lines": "p6_filter_max_lines",
    "prob_high_threshold": "p6_filter_prob_high_threshold",
    "prob_low_threshold": "p6_filter_prob_low_threshold",
    "bright_threshold_for_led_prob": "p6_filter_bright_threshold",
    "line_sampling_width": "p6_filter_line_sampling_width",
    "line_sampling_num_samples": "p6_filter_line_sampling_num_samples",
}


def rename_in_content(content, old_name, new_name):
    """
    Rename a parameter in file content.
    Handles various patterns:
    - Dictionary keys: "old_name": or 'old_name':
    - List items: ["old_name"] or ['old_name']
    - Function parameters: old_name=
    - Variable names: old_name:
    - Comments and strings containing the name
    """
    # Pattern 1: Dictionary keys with double quotes
    content = re.sub(
        rf'"{re.escape(old_name)}"(\s*):',
        rf'"{new_name}"\1:',
        content
    )

    # Pattern 2: Dictionary keys with single quotes
    content = re.sub(
        rf"'{re.escape(old_name)}'(\s*):",
        rf"'{new_name}'\1:",
        content
    )

    # Pattern 3: List items with double quotes
    content = re.sub(
        rf'"\["{re.escape(old_name)}"\]"',
        rf'"["{new_name}"]"',
        content
    )

    # Pattern 4: List items with single quotes
    content = re.sub(
        rf"'\['{re.escape(old_name)}'\]'",
        rf"'['{new_name}']'",
        content
    )

    # Pattern 5: TypedDict field definitions (with type annotations)
    content = re.sub(
        rf'\b{re.escape(old_name)}(\s*):',
        rf'{new_name}\1:',
        content
    )

    # Pattern 6: Function parameter names
    content = re.sub(
        rf'\b{re.escape(old_name)}(\s*)=',
        rf'{new_name}\1=',
        content
    )

    # Pattern 7: config["old_name"] or config['old_name']
    content = re.sub(
        rf'\["{re.escape(old_name)}"\]',
        rf'["{new_name}"]',
        content
    )
    content = re.sub(
        rf"\['{re.escape(old_name)}'\]",
        rf"['{new_name}']",
        content
    )

    return content


def process_file(file_path, renames):
    """Process a single file and apply all renames."""
    if not os.path.exists(file_path):
        print(f"  ⚠ File not found: {file_path}")
        return False

    # Read file
    with open(file_path, 'r') as f:
        original_content = f.read()

    # Apply all renames
    modified_content = original_content
    changes_made = []

    for old_name, new_name in renames.items():
        # Check if old_name exists in the file
        if old_name in modified_content:
            modified_content = rename_in_content(modified_content, old_name, new_name)
            if modified_content != original_content:
                changes_made.append((old_name, new_name))
                original_content = modified_content

    # Write back if changes were made
    if changes_made:
        with open(file_path, 'w') as f:
            f.write(modified_content)
        return True

    return False


def main():
    """Main renaming function."""
    workspace_path = "/Users/pepijnvanderklei/School/VFAR/vfar_ws/research"

    # Files to process
    files_to_process = [
        "pipeline.py",
        "evaluate.py",
    ]

    print("=" * 80)
    print("PARAMETER RENAMING")
    print("=" * 80)
    print(f"Total renames to apply: {len(RENAMES)}")
    print()

    # Process each file
    for file_name in files_to_process:
        file_path = os.path.join(workspace_path, file_name)
        print(f"Processing: {file_name}")

        if process_file(file_path, RENAMES):
            print(f"  ✓ Updated {file_name}")
        else:
            print(f"  ○ No changes needed in {file_name}")

    print()
    print("=" * 80)
    print("✓ RENAMING COMPLETED!")
    print("=" * 80)
    print()
    print("Summary of renames:")
    for old_name, new_name in sorted(RENAMES.items()):
        print(f"  {old_name:40s} -> {new_name}")


if __name__ == "__main__":
    main()
