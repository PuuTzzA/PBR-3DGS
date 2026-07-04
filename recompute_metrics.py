#!/usr/bin/env python3
"""Recompute relighting metrics for all models without re-rendering.

This reads existing *_relight.png files from all_relighting/ and recalculates
both raw and scale-invariant metrics (PSNR, SSIM, MSE).
"""
import os
import sys
import subprocess

BASE = "outputs/new_experiments_try_4_presentation"
MODELS = {
    "Baseline": f"{BASE}/lego_baseline_no_prior",
    "with Priors": f"{BASE}/lego_albedo_warmup_zncc",
    "with Diffusion Priors": f"{BASE}/lego_diffusion_zncc"
}
DATASET_PATH = "data/datasets_with_priors/lego"
HDRIS = ["city", "courtyard", "snow", "fireplace", "night", "forest"]

# Use the python interpreter running this script, or fallback to the specific conda env
if "miniconda3/envs/gir" in sys.executable:
    python_executable = sys.executable
else:
    conda_python = "/home/ljochim/miniconda3/envs/gir/bin/python"
    python_executable = conda_python if os.path.exists(conda_python) else sys.executable

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir:
        os.chdir(script_dir)

    print("Recomputing relighting metrics for all models...")
    print(f"Using python interpreter: {python_executable}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"HDRIs: {HDRIS}\n")

    for name, model_path in MODELS.items():
        print("=" * 80)
        print(f"Recomputing metrics for: {name}")
        print(f"Model Path: {model_path}")
        print("=" * 80)

        if not os.path.exists(model_path):
            print(f"ERROR: Model directory does not exist: {model_path}\n")
            continue

        cmd = [
            python_executable,
            "relight_all.py",
            "-m", model_path,
            "-d", DATASET_PATH,
            "--hdris", *HDRIS,
            "--recompute_metrics"
        ]

        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Finished metrics recomputation for: {name}\n")
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

    print("All models processed.")

if __name__ == "__main__":
    main()
