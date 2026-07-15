#!/usr/bin/env python3
"""
Utility script to automate PBR-3DGS model relighting under all available HDR environment maps (HDRIs) in a dataset.

Usage:
    /home/ljochim/miniconda3/envs/gir/bin/python relight_all.py -m <model_path> -d <dataset_path> [options]

Arguments:
    -m, --model_path      Path to the model outputs folder (e.g. outputs/experiment_with_all_4_priors_2/soft_priors_all_materials).
    -d, --dataset_path    Path to the source dataset directory (e.g. data/datasets_with_priors/lego).
    -i, --iteration       (Optional) Model iteration to load (defaults to the highest found in the model folder).
    --hdris_dir           (Optional) Custom directory containing .hdr environment maps (defaults to <dataset_path>/hdris).

Outputs:
    Saves all outputs directly inside the model's folder at `<model_path>/all_relighting/`:
    1. `<hdri_name>/` - Subfolder for each HDRI containing:
        - `<view_idx>_relight.png` - The model's relighted output, aligned to the ground truth using optimal scale factor.
        - `<view_idx>_render.png` - The corresponding ground truth image (if available in dataset's `test/rgba_<hdri_name>`).
    2. `relighting_metrics.json` - JSON file mapping each HDRI to its average scale factor (gain) and scale-invariant metrics (PSNR, SSIM, MSE).
"""

import os
import sys
import json
import glob
import re
import argparse
import shutil
import subprocess
import numpy as np
from PIL import Image

# Configure path so we can import packages from GIR if needed
sys.path.insert(0, os.path.abspath("GIR"))

def parse_cfg_args(cfg_path):
    """Parse cfg_args Namespace file into a dict of arguments."""
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r") as f:
        text = f.read().strip()
    
    if text.startswith("Namespace("):
        try:
            class Namespace:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
            ns = eval(text, {"Namespace": Namespace})
            return ns.__dict__
        except Exception:
            pass
            
    pairs = re.findall(r"(\w+)\s*=\s*(True|False|None|\[.*?\]|[^,\)]+)", text)
    result = {}
    for k, v in pairs:
        v = v.strip()
        if v == "True":
            result[k] = True
        elif v == "False":
            result[k] = False
        elif v == "None":
            result[k] = None
        elif v.startswith("'") or v.startswith('"'):
            result[k] = v[1:-1]
        else:
            try:
                if "." in v:
                    result[k] = float(v)
                else:
                    result[k] = int(v)
            except ValueError:
                result[k] = v
    return result

def find_highest_iteration(model_path):
    """Scan the model folder for checkpoints and point clouds to find the highest iteration."""
    iters = []
    # Check checkpoints
    for f in glob.glob(os.path.join(model_path, "chkpnt*.pth")):
        match = re.search(r"chkpnt(\d+)\.pth", os.path.basename(f))
        if match:
            iters.append(int(match.group(1)))
    # Check point clouds
    for d in glob.glob(os.path.join(model_path, "point_cloud", "iteration_*")):
        match = re.search(r"iteration_(\d+)", os.path.basename(d))
        if match:
            iters.append(int(match.group(1)))
    if iters:
        return max(iters)
    return None

def check_and_restore_ply(model_path, iteration):
    """Check if point_cloud.ply exists for an iteration, and if not, restore it from the checkpoint."""
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if os.path.exists(ply_path):
        print(f"Found existing PLY at: {ply_path}")
        return True
        
    checkpoint_path = os.path.join(model_path, f"chkpnt{iteration}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Neither PLY file nor checkpoint found at iteration {iteration} in {model_path}")
        return False
        
    print(f"PLY file missing for iteration {iteration} in {model_path}. Reconstructing from checkpoint: {checkpoint_path}...")
    try:
        import torch
        from scene.gaussian_model import GaussianModel
        
        # Load checkpoint parameters
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_args = checkpoint[0]
        
        if not torch.cuda.is_available():
            print("ERROR: CUDA is not available. Cannot run GaussianModel reconstruction.")
            return False
            
        gaussians = GaussianModel(sh_degree=3)
        
        # Assign parameters directly, moving them to GPU
        gaussians.active_sh_degree = model_args[0]
        gaussians._xyz = torch.nn.Parameter(model_args[1].cuda())
        gaussians._features_dc = torch.nn.Parameter(model_args[2].cuda())
        gaussians._features_rest = torch.nn.Parameter(model_args[3].cuda())
        gaussians._scaling = torch.nn.Parameter(model_args[4].cuda())
        gaussians._rotation = torch.nn.Parameter(model_args[5].cuda())
        gaussians._opacity = torch.nn.Parameter(model_args[6].cuda())
        gaussians._albedo_init = torch.nn.Parameter(model_args[7].cuda())
        gaussians._metallic_init = torch.nn.Parameter(model_args[8].cuda())
        gaussians._roughness_init = torch.nn.Parameter(model_args[9].cuda())
        
        # Save reconstructed PLY file
        os.makedirs(os.path.dirname(ply_path), exist_ok=True)
        gaussians.save_ply(ply_path)
        print(f"Reconstruction successful! Saved PLY to: {ply_path}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to reconstruct PLY file from checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_image_as_float_np(path, target_size=None):
    """Load image as float32 numpy array [H,W,3] in [0,1], optionally resizing to target_size (width, height)."""
    img = Image.open(path).convert("RGB")
    if target_size is not None and img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0

def compute_psnr_np(img1, img2):
    """PSNR between two float32 images in [0,1]."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return -10.0 * np.log10(mse)

def compute_ssim_np(img1, img2, window_size=11):
    """Simplified SSIM (per-channel mean) between two float32 images [H,W,3] in [0,1]."""
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        print("ERROR: scipy is required for SSIM calculation.")
        return 0.0
    C1, C2 = 0.01**2, 0.03**2
    ssim_vals = []
    for c in range(img1.shape[2]):
        a, b = img1[:,:,c], img2[:,:,c]
        mu_a = uniform_filter(a, window_size)
        mu_b = uniform_filter(b, window_size)
        sig_a2 = uniform_filter(a*a, window_size) - mu_a*mu_a
        sig_b2 = uniform_filter(b*b, window_size) - mu_b*mu_b
        sig_ab = uniform_filter(a*b, window_size) - mu_a*mu_b
        num = (2*mu_a*mu_b + C1) * (2*sig_ab + C2)
        den = (mu_a**2 + mu_b**2 + C1) * (sig_a2 + sig_b2 + C2)
        ssim_vals.append(np.mean(num / den))
    return np.mean(ssim_vals)

def align_image_np(render, gt):
    """Compute optimal scalar exposure gain to align render to GT, clipping to [0,1]."""
    num = np.sum(render * gt)
    den = np.sum(render * render)
    gain = num / max(den, 1e-8)
    aligned = np.clip(render * gain, 0.0, 1.0)
    return gain, aligned

def run_relighting(model_path, hdri_name, hdri_path, iteration, dataset_path, white_background=True, hdr_rotation=True):
    """Invoke GIR/render.py for relighting rendering via subprocess."""
    model_path_abs = os.path.abspath(model_path)
    hdri_path_abs = os.path.abspath(hdri_path)
    dataset_path_abs = os.path.abspath(dataset_path)
    
    cmd = [
        sys.executable,
        "render.py",
        "-m", model_path_abs,
        "-s", dataset_path_abs,
        "--skip_train",
        "--save_name", hdri_name,
        "--environment_texture", hdri_path_abs,
        "--render_relight",
        "--iteration", str(iteration)
    ]
    if white_background:
        cmd.append("-w")
    if hdr_rotation:
        cmd.append("--hdr_rotation")
        
    print(f"Running relighting process for {hdri_name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="GIR", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"ERROR: Relighting execution failed for {hdri_name}:")
        print(result.stderr)
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Evaluate PBR-3DGS Model under all HDRIs in a dataset.")
    parser.add_argument("-m", "--model_path", required=True, type=str, help="Path to the model directory in outputs/")
    parser.add_argument("-d", "--dataset_path", required=True, type=str, help="Path to the dataset directory in datasets_with_priors/")
    parser.add_argument("-i", "--iteration", default=None, type=int, help="Model iteration to load (defaults to highest found)")
    parser.add_argument("--hdris_dir", default=None, type=str, help="Directory containing .hdr environment maps (defaults to dataset_path/hdris)")
    parser.add_argument("--hdris", nargs="+", default=None, type=str, help="Only process these HDRIs (names without .hdr extension, e.g. city courtyard snow)")
    parser.add_argument("--recompute_metrics", action="store_true", help="Recalculate metrics for already-processed HDRIs without re-rendering")
    args = parser.parse_args()

    # Resolve paths
    model_path = os.path.abspath(args.model_path)
    dataset_path = os.path.abspath(args.dataset_path)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model path does not exist: {model_path}")
        sys.exit(1)
        
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    # Load configuration arguments of model to inherit flags
    cfg_args = parse_cfg_args(os.path.join(model_path, "cfg_args"))
    white_background = cfg_args.get("white_background", True)
    hdr_rotation = cfg_args.get("hdr_rotation", True)
    print(f"Inherited Config from Model: white_background={white_background}, hdr_rotation={hdr_rotation}")

    # Determine iteration
    if args.iteration is not None:
        iteration = args.iteration
    else:
        iteration = find_highest_iteration(model_path)
        if iteration is None:
            print(f"ERROR: Could not automatically detect any trained iteration in {model_path}")
            sys.exit(1)
    
    print(f"Using model iteration: {iteration}")

    # Verify and restore PLY if needed
    if not check_and_restore_ply(model_path, iteration):
        print("ERROR: Point cloud could not be verified or reconstructed.")
        sys.exit(1)

    # Locate HDRI directory
    hdris_dir = os.path.abspath(args.hdris_dir) if args.hdris_dir else os.path.join(dataset_path, "hdris")
    if not os.path.exists(hdris_dir):
        print(f"ERROR: HDRI directory not found at: {hdris_dir}")
        sys.exit(1)

    # Find HDRI maps (optionally filtered by --hdris)
    if args.hdris:
        hdri_files = []
        for name in args.hdris:
            path = os.path.join(hdris_dir, f"{name}.hdr")
            if os.path.exists(path):
                hdri_files.append(path)
            else:
                print(f"WARNING: HDRI not found, skipping: {path}")
        hdri_files = sorted(hdri_files)
    else:
        hdri_files = sorted(glob.glob(os.path.join(hdris_dir, "*.hdr")))
    if not hdri_files:
        print(f"ERROR: No .hdr files found in: {hdris_dir}")
        sys.exit(1)
    
    print(f"Processing {len(hdri_files)} HDRI files: {[os.path.splitext(os.path.basename(f))[0] for f in hdri_files]}")

    # Create target directory
    all_relighting_dir = os.path.join(model_path, "all_relighting")
    os.makedirs(all_relighting_dir, exist_ok=True)
    print(f"Relighting outputs will be saved to: {all_relighting_dir}")

    # Load existing metrics summary if it exists to allow resuming
    metrics_json_path = os.path.join(all_relighting_dir, "relighting_metrics.json")
    metrics_summary = {}
    if os.path.exists(metrics_json_path):
        try:
            with open(metrics_json_path, "r") as f:
                metrics_summary = json.load(f)
            print(f"Loaded existing metrics for {len(metrics_summary)} HDRIs from {metrics_json_path}")
        except Exception as e:
            print(f"Could not load existing metrics file: {e}")

    for idx_hdr, hdri_path in enumerate(hdri_files):
        hdri_filename = os.path.basename(hdri_path)
        hdri_name = os.path.splitext(hdri_filename)[0]

        # Create subfolder for this HDRI
        hdri_output_dir = os.path.join(all_relighting_dir, hdri_name)
        os.makedirs(hdri_output_dir, exist_ok=True)

        # Check if already processed (skip only if not recomputing metrics)
        existing_relight_pngs = sorted(glob.glob(os.path.join(hdri_output_dir, "*_relight.png")))
        already_rendered = len(existing_relight_pngs) > 0

        if already_rendered and not args.recompute_metrics:
            if hdri_name in metrics_summary:
                print(f"\n[{idx_hdr+1}/{len(hdri_files)}] HDRI: {hdri_name} already processed. Skipping.")
                continue

        print(f"\n[{idx_hdr+1}/{len(hdri_files)}] Processing HDRI: {hdri_name} ...")

        # Determine where to read rendered images from
        if already_rendered and args.recompute_metrics:
            # Recompute mode: use existing _relight.png files from the output dir
            print(f"Recomputing metrics from existing renders in {hdri_output_dir}")
            render_files = existing_relight_pngs
            need_save_images = False
        else:
            # Normal mode: run rendering
            success = run_relighting(model_path, hdri_name, hdri_path, iteration, dataset_path, white_background, hdr_rotation)
            if not success:
                print(f"Skipping HDRI {hdri_name} due to rendering failure.")
                continue

            renders_dir = os.path.join(model_path, f"test_{hdri_name}", f"ours_{iteration}", "renders")
            render_files = sorted(glob.glob(os.path.join(renders_dir, "*.png")))

            if not render_files:
                print(f"Warning: Rendered images not found in {renders_dir}")
                continue
            need_save_images = True

        # Look for Ground Truth folder
        gt_dir = os.path.join(dataset_path, "test", f"rgba_{hdri_name}")
        if not os.path.exists(gt_dir):
            # Fallback to test/rgba or main folder rgba if split is different
            gt_dir = os.path.join(dataset_path, f"rgba_{hdri_name}")

        has_gt = os.path.exists(gt_dir)
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png"))) if has_gt else []

        if has_gt and len(gt_files) > 0:
            print(f"Found Ground Truth under: {gt_dir}. Performing alignment and metrics calculation...")
            
            gains = []
            raw_psnrs = []
            raw_ssims = []
            raw_mses = []
            si_psnrs = []
            si_ssims = []
            si_mses = []

            # Match renders and GTs by sorted order
            num_views = min(len(render_files), len(gt_files))
            for idx in range(num_views):
                r_file = render_files[idx]
                g_file = gt_files[idx]

                # Load images
                render_np = load_image_as_float_np(r_file)
                h_r, w_r = render_np.shape[:2]
                gt_np = load_image_as_float_np(g_file, target_size=(w_r, h_r))

                # Raw (unaligned) metrics
                raw_psnrs.append(compute_psnr_np(render_np, gt_np))
                raw_ssims.append(compute_ssim_np(render_np, gt_np))
                raw_mses.append(float(np.mean((render_np - gt_np) ** 2)))

                # Compute scale-invariant alignment gain
                gain, aligned_np = align_image_np(render_np, gt_np)
                gains.append(gain)

                # Scale-invariant metrics
                si_psnrs.append(compute_psnr_np(aligned_np, gt_np))
                si_ssims.append(compute_ssim_np(aligned_np, gt_np))
                si_mses.append(float(np.mean((aligned_np - gt_np) ** 2)))

                if need_save_images:
                    # Save aligned render and ground truth to target HDRI folder
                    base_name = os.path.splitext(os.path.basename(r_file))[0]
                    relight_save_path = os.path.join(hdri_output_dir, f"{base_name}_relight.png")
                    render_save_path = os.path.join(hdri_output_dir, f"{base_name}_render.png")
                    Image.fromarray((aligned_np * 255).astype(np.uint8)).save(relight_save_path)
                    shutil.copy(g_file, render_save_path)

            # Store summary statistics
            avg_gain = float(np.mean(gains))
            avg_raw_psnr = float(np.mean(raw_psnrs))
            avg_raw_ssim = float(np.mean(raw_ssims))
            avg_raw_mse = float(np.mean(raw_mses))
            avg_si_psnr = float(np.mean(si_psnrs))
            avg_si_ssim = float(np.mean(si_ssims))
            avg_si_mse = float(np.mean(si_mses))

            print(f"Average scale factor: {avg_gain:.4f}")
            print(f"Raw metrics              -> PSNR: {avg_raw_psnr:.2f} dB | SSIM: {avg_raw_ssim:.4f} | MSE: {avg_raw_mse:.6f}")
            print(f"Scale-invariant metrics  -> PSNR: {avg_si_psnr:.2f} dB | SSIM: {avg_si_ssim:.4f} | MSE: {avg_si_mse:.6f}")

            metrics_summary[hdri_name] = {
                "scale_factor": avg_gain,
                "raw_psnr": avg_raw_psnr,
                "raw_ssim": avg_raw_ssim,
                "raw_mse": avg_raw_mse,
                "scale_invariant_psnr": avg_si_psnr,
                "scale_invariant_ssim": avg_si_ssim,
                "scale_invariant_mse": avg_si_mse
            }
        else:
            print(f"No Ground Truth found for HDRI: {hdri_name}. Saving raw renders directly...")
            if need_save_images:
                for r_file in render_files:
                    base_name = os.path.splitext(os.path.basename(r_file))[0]
                    relight_save_path = os.path.join(hdri_output_dir, f"{base_name}_relight.png")
                    shutil.copy(r_file, relight_save_path)
            
            metrics_summary[hdri_name] = {
                "scale_factor": None,
                "raw_psnr": None,
                "raw_ssim": None,
                "raw_mse": None,
                "scale_invariant_psnr": None,
                "scale_invariant_ssim": None,
                "scale_invariant_mse": None
            }

        # Write summary metrics JSON file incrementally after each HDRI
        with open(metrics_json_path, "w") as f:
            json.dump(metrics_summary, f, indent=4)

        # Remove temporary rendered folder generated by render.py for this HDRI to save space
        if need_save_images:
            temp_dir = os.path.join(model_path, f"test_{hdri_name}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    print(f"\nSaved scale-invariant relighting values to: {metrics_json_path}")
    print("\nHDRI relighting evaluation completed successfully!")

if __name__ == "__main__":
    main()
