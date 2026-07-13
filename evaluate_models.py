#!/usr/bin/env python3
"""Run post-hoc model evaluation (albedo + normal + relighting) for all configured models.

This is a thin wrapper that invokes evaluate_model.py for each model,
mirroring the pattern of relight_sequence.py.
"""
import os
import sys
import subprocess
import argparse
import json

BASE = "outputs/poster_results/hotdog"
MODELS = {
    "Baseline": f"{BASE}/hotdog_baseline_no_prior",
    "with Diffusion Priors": f"{BASE}/hotdog_diff_zncc_zncc",
    "with GT Priors": f"{BASE}/hotdog_gt_zncc_zncc_neu"
}
DATASET_PATH = "data/datasets_with_priors/hotdog_data"
HDRIS = ["snow", "fireplace", "night"]

# Use the python interpreter running this script, or fallback to the specific conda env
if "miniconda3/envs/gir" in sys.executable:
    python_executable = sys.executable
else:
    conda_python = "/home/ljochim/miniconda3/envs/gir/bin/python"
    python_executable = conda_python if os.path.exists(conda_python) else sys.executable

def main():
    arg_parser = argparse.ArgumentParser(
        description="Run post-hoc model evaluation for all configured models.")
    arg_parser.add_argument("--lli", action="store_true",
                            help="Enable light_linear_indirect for relighting "
                                 "evaluation (only for Diffusion/GT Priors models)")
    cli_args = arg_parser.parse_args()

    # Models that support LLI (have PBR decomposition with indirect lighting)
    LLI_MODELS = {"with Diffusion Priors", "with GT Priors"}

    # Make sure we run in the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir:
        os.chdir(script_dir)

    print("Starting PBR-3DGS sequential model evaluation...")
    print(f"Using python interpreter: {python_executable}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"HDRIs: {HDRIS}")
    if cli_args.lli:
        print(f"LLI:   enabled (for Diffusion/GT Priors models)")
    print()

    results = {}  # name -> metrics dict

    for name, model_path in MODELS.items():
        print("=" * 80)
        print(f"Evaluating model: {name}")
        print(f"Model Path: {model_path}")
        print("=" * 80)

        if not os.path.exists(model_path):
            print(f"ERROR: Model directory does not exist: {model_path}\n")
            continue

        cmd = [
            python_executable,
            "evaluate_model.py",
            "-m", model_path,
            "-s", DATASET_PATH,
            "--hdris", *HDRIS
        ]

        # Add --lli flag for models that support it
        if cli_args.lli and name in LLI_MODELS:
            cmd.append("--lli")

        print(f"Executing: {' '.join(cmd)}")
        success = False
        try:
            # Run the command and stream output directly to console
            subprocess.run(cmd, check=True)
            print(f"Finished evaluation for: {name}\n")
            success = True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Command failed for {name} with exit code {e.returncode}\n")
        except KeyboardInterrupt:
            print(f"\nInterrupted! Skipping model: {name}")
            print("(Press Ctrl+C again within 2s to exit entirely)\n")
            import time
            try:
                time.sleep(2)
            except KeyboardInterrupt:
                print("\nExiting.")
                sys.exit(1)

        # Collect results from the per-model JSON (only after success)
        if success:
            metrics_path = os.path.join(model_path, "evaluation_metrics.json")
            if os.path.isfile(metrics_path):
                with open(metrics_path) as f:
                    results[name] = json.load(f)

    # ── Combined summary ─────────────────────────────────────────────────
    if results:
        # Save combined JSON
        summary_path = os.path.join(BASE, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Combined results saved to: {summary_path}")

        # Print table for easy copy-paste
        # Gather all metric keys (excluding 'model') in a stable order
        all_keys = []
        for row in results.values():
            for k in row:
                if k != "model" and k not in all_keys:
                    all_keys.append(k)

        col_width = 14
        header = f"{'Model':<30}" + "".join(f"{k:>{col_width}}" for k in all_keys)
        print("\n" + "=" * len(header))
        print("COMBINED RESULTS")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for name, row in results.items():
            vals = ""
            for k in all_keys:
                v = row.get(k)
                if v is None:
                    vals += f"{'—':>{col_width}}"
                else:
                    vals += f"{v:>{col_width}.4f}"
            print(f"{name:<30}{vals}")
        print("=" * len(header))
    else:
        print("No results collected.")

    print("All models evaluated.")

if __name__ == "__main__":
    main()
