import os
import cv2
import argparse
import numpy as np
from typing import List, Tuple
from itertools import product
from tqdm import tqdm
import json
import shutil

from pipeline import (
    LineDetectionPipeline,
    PipelineConfig,
    MetricsResult,
)


def create_grid_search_configs() -> List[PipelineConfig]:
    """
    Create grid search configurations.

    Optimized for detecting multiple LED light strips on ceiling.
    """
    configs = []

    # Define search space - comprehensive preprocessing pipeline
    colorspaces = ["hsv"]  # HSV-V works best for brightness
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
    edge_methods = ["canny"]  # Canny is best
    line_methods = ["hough"]  # Focus on Hough for now
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
    for (
        colorspace,
        brightness_threshold,
        use_bg_norm,
        use_clahe,
        use_adaptive,
        roi_top_ratio,
        kernel_size,
        iterations,
        use_closing,
        use_overlay,
        edge_method,
        line_method,
        merge_lines,
    ) in product(
        colorspaces,
        brightness_thresholds,
        use_background_norms,
        use_clahes,
        use_adaptives,
        roi_top_ratios,
        morph_kernel_sizes,
        morph_iterations,
        use_closings,
        use_overlays,
        edge_methods,
        line_methods,
        merge_lines_options,
    ):
        # Base configuration
        config = {
            "colorspace": colorspace,
            "brightness_threshold": brightness_threshold,
            "use_background_norm": use_bg_norm,
            "use_clahe": use_clahe,
            "use_adaptive": use_adaptive,
            "roi_top_ratio": roi_top_ratio,
            "morph_kernel_size": kernel_size,
            "morph_iterations": iterations,
            "use_closing": use_closing,
            "use_overlay": use_overlay,
            "edge_method": edge_method,
            "line_method": line_method,
            "merge_lines": merge_lines,
        }

        # Add method-specific parameters
        if edge_method == "canny":
            for canny_low, canny_high, blur_kernel in product(
                canny_lows, canny_highs, blur_kernels
            ):
                config_copy = config.copy()
                config_copy["canny_low"] = canny_low
                config_copy["canny_high"] = canny_high
                config_copy["blur_kernel"] = blur_kernel

                if line_method == "hough":
                    for hough_threshold, min_length, max_gap in product(
                        hough_thresholds, min_line_lengths, max_line_gaps
                    ):
                        config_final = config_copy.copy()
                        config_final["hough_threshold"] = hough_threshold
                        config_final["min_line_length"] = min_length
                        config_final["max_line_gap"] = max_gap
                        configs.append(config_final)
                else:
                    configs.append(config_copy)

    print(f"Generated {len(configs)} configurations for grid search")
    return configs


def generate_config_name(config: PipelineConfig) -> str:
    """Generate a human-readable name encoding all important config parameters."""
    # Build preprocessing flags
    flags = []
    if config.get("use_background_norm"):
        flags.append("bg")
    if config.get("use_clahe"):
        flags.append("cl")
    if config.get("use_adaptive"):
        flags.append("ad")
    if config.get("blur_kernel", 0) > 0:
        flags.append(f"b{config.get('blur_kernel', 0)}")
    if config.get("merge_lines"):
        flags.append("mg")
    flags_str = "-".join(flags) if flags else "plain"

    parts = [
        config["colorspace"],
        f"thr{config['brightness_threshold']}",
        f"k{config['morph_kernel_size']}i{config['morph_iterations']}",
        "ovr" if config["use_overlay"] else "msk",
        config["edge_method"],
        config["line_method"],
    ]

    # Add edge/line detection params
    if config.get("canny_low"):
        parts.append(f"cl{config.get('canny_low')}")
    if config.get("canny_high"):
        parts.append(f"ch{config.get('canny_high')}")
    if config.get("hough_threshold"):
        parts.append(f"h{config.get('hough_threshold')}")
    if config.get("min_line_length"):
        parts.append(f"ml{config.get('min_line_length')}")

    # Add ROI if used
    if config.get("roi_top_ratio", 0) > 0:
        parts.append(f"roi{int(config.get('roi_top_ratio', 0)*100)}")

    # Add preprocessing flags at the end
    parts.append(flags_str)

    return "_".join(str(p) for p in parts if p)


def save_results(
    output_folder: str,
    config_id: int,
    config: PipelineConfig,
    metrics: MetricsResult,
    steps: List[Tuple[str, np.ndarray]],
    pipeline: LineDetectionPipeline,
):
    """Save all images for a configuration."""
    # Create folder for this configuration
    config_name = generate_config_name(config)
    folder_path = os.path.join(output_folder, f"{config_id:03d}_{config_name}")
    os.makedirs(folder_path, exist_ok=True)

    # Save all intermediate steps
    for step_name, image in steps:
        cv2.imwrite(os.path.join(folder_path, f"{step_name}.png"), image)

    # Save configuration and metrics
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
    }
    with open(os.path.join(folder_path, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    return folder_path


def draw_target_lines(pipeline: LineDetectionPipeline, save_path: str):
    """Draw target lines on original image and save visualization."""
    if not pipeline.targets or pipeline.original_color is None:
        return

    result = pipeline.original_color.copy()
    target_lines = pipeline.targets.get("target_lines", [])

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
        "input_image",
        nargs="?",
        default="field.png",
        type=str,
        help="Path to the input image (default: field.png)",
    )
    parser.add_argument(
        "output_folder",
        nargs="?",
        default=os.path.join("output", "linedetect"),
        type=str,
        help="Path to save results (default: output/linedetect)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top configurations to save (default: 10)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="targets.json",
        help="Path to targets JSON file (default: targets.json)",
    )
    parser.add_argument(
        "--preview-targets",
        action="store_true",
        help="Only generate target_lines_reference.png and exit (for quick editing)",
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
    image = cv2.imread(args.input_image)
    if image is None:
        print(f"Error: Could not load image from {args.input_image}")
        return

    # Load targets if available
    targets = None
    if os.path.exists(args.targets):
        print(f"Loading targets from: {args.targets}")
        with open(args.targets, "r") as f:
            targets = json.load(f)
    else:
        print("No targets file found, using generic scoring")

    pipeline = LineDetectionPipeline(image, targets)

    # Draw target lines visualization if targets are provided
    if targets:
        target_viz_path = os.path.join(args.output_folder, "target_lines_reference.png")
        draw_target_lines(pipeline, target_viz_path)

    # If preview mode, exit after drawing targets
    if args.preview_targets:
        print("\nPreview mode: Target visualization complete. Exiting.")
        return

    # Generate configurations
    print("\n" + "=" * 80)
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
            # Create a sub-progress bar for this configuration
            with tqdm(
                total=6,
                desc=f"Config {i+1}",
                position=1,
                leave=False,
                bar_format="{desc}: {bar}| {n_fmt}/{total_fmt}",
            ) as subpbar:
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

            results.append(
                {"config_id": i, "config": config, "metrics": metrics, "steps": steps}
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
                args.output_folder,
                rank + 1,
                result["config"],
                result["metrics"],
                result["steps"],
                pipeline,
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
    summary_path = os.path.join(args.output_folder, "summary.json")
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
            total_targets = (
                len(pipeline.targets["target_lines"]) if pipeline.targets else 0
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
    print(f"Top {args.top_k} results saved to: {args.output_folder}")


if __name__ == "__main__":
    main()
