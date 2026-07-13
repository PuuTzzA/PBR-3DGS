#!/usr/bin/env python3
"""
Script to average metrics from predefined JSON evaluation summary files:
- outputs/poster_results/lego/evaluation_summary.json
- outputs/poster_results/hotdog/evaluation_summary.json
"""

import os
import sys
import json
import numpy as np

# Predefined input paths
INPUT_FILES = [
    "outputs/poster_results/lego/evaluation_summary.json",
    "outputs/poster_results/hotdog/evaluation_summary.json"
]

OUTPUT_FILE = "outputs/poster_results/averaged_summary.json"

def main():
    # Filter and validate input files
    valid_files = []
    for f in INPUT_FILES:
        if os.path.isfile(f):
            valid_files.append(f)
        else:
            print(f"Warning: Predefined file not found: {f}", file=sys.stderr)

    if not valid_files:
        print("Error: No valid JSON files found to average.", file=sys.stderr)
        sys.exit(1)

    print(f"Averaging {len(valid_files)} JSON files:")
    for f in valid_files:
        print(f"  - {f}")
    print()

    # Structure to hold metrics:
    # model_name -> metric_name -> list of values
    aggregated_metrics = {}

    for file_path in valid_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading/parsing {file_path}: {e}", file=sys.stderr)
            continue

        # Each summary JSON is a dict: { model_name: { metric_name: value } }
        for model_name, metrics in data.items():
            if model_name not in aggregated_metrics:
                aggregated_metrics[model_name] = {}

            for k, v in metrics.items():
                # Skip non-numeric values (like model paths or names)
                if k == "model" or not isinstance(v, (int, float)):
                    continue

                if k not in aggregated_metrics[model_name]:
                    aggregated_metrics[model_name][k] = []
                aggregated_metrics[model_name][k].append(v)

    # Calculate mean and std
    averaged_results = {}
    for model_name, metrics_dict in aggregated_metrics.items():
        averaged_results[model_name] = {}
        for metric_name, values in metrics_dict.items():
            if len(values) > 0:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values)) if len(values) > 1 else 0.0
                averaged_results[model_name][metric_name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "n": len(values),
                }

    # Save to output JSON file (matching the input structure but with mean values)
    output_data = {}
    for model_name, metrics_dict in averaged_results.items():
        output_data[model_name] = {}
        for metric_name, stats in metrics_dict.items():
            output_data[model_name][metric_name] = stats["mean"]

    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Averaged results saved to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)

    # Print a formatted summary table
    # Gather all unique metric keys
    all_keys = []
    for model_name, metrics in averaged_results.items():
        for k in metrics:
            if k not in all_keys:
                all_keys.append(k)
    all_keys.sort()

    col_width = 16
    header = f"{'Model':<30}" + "".join(f"{k:>{col_width}}" for k in all_keys)
    print("\n" + "=" * len(header))
    print(f"AVERAGED RESULTS (N = {len(valid_files)})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for model_name, metrics_dict in averaged_results.items():
        row_str = f"{model_name:<30}"
        for k in all_keys:
            stats = metrics_dict.get(k)
            if stats is None:
                row_str += f"{'—':>{col_width}}"
            else:
                row_str += f"{stats['mean']:>{col_width}.4f}"
        print(row_str)
    print("=" * len(header))


if __name__ == "__main__":
    main()
