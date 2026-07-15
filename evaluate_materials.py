#!/usr/bin/env python3
"""
Post-hoc evaluation of decomposed material properties (albedo, normal) for a
fully trained PBR-3DGS model.

This script loads a trained model checkpoint, renders the decomposed albedo
and normal maps for every test view, compares them against the dataset's ground
truth, and produces:

  1. Per-view rendered images saved under
       <model_path>/<material>_evaluation_aftermath/
  2. A JSON file with aggregate metrics.

Albedo metrics
--------------
A **global per-channel 3D scale vector** [s_r, s_g, s_b] is computed across
ALL test views simultaneously (closed-form least squares) and applied uniformly
to every pixel of every rendered albedo.  The aligned albedo is then evaluated
with PSNR. Raw (unaligned) PSNR and SSIM are also reported for reference.

Normal metrics
--------------
Mean angular error (degrees) between predicted and ground-truth normals.

Usage:
    cd PBR-3DGS
    python evaluate_materials.py \
        -m <model_path> \
        -s <dataset_path> \
        [--iteration <N>] \
        [--albedo_gt_dir albedo_gt] \
        [--normal_gt_dir normal_gt]
"""

import os
import sys
import json
import math

# ---------------------------------------------------------------------------
# We need access to GIR's own packages (scene, gaussian_renderer, …).
# The script lives in PBR-3DGS/, one level above GIR/.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GIR_DIR = os.path.join(SCRIPT_DIR, "GIR")
sys.path.insert(0, GIR_DIR)

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from utils.loss_utils import ssim as compute_ssim


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def psnr_torch(img, gt):
    """PSNR between two [C,H,W] float tensors in [0,1]."""
    mse = torch.mean((img - gt) ** 2)
    if mse < 1e-10:
        return float("inf")
    return (-10.0 * torch.log10(mse)).item()


def angular_error_torch(pred_normal, gt_normal):
    """Mean angular error in degrees between two [3,H,W] normal maps.

    Both maps are assumed to be in [0,1] range (i.e. (n+1)/2 encoded) and are
    un-packed to [-1,1] before comparison.
    """
    pred = pred_normal * 2.0 - 1.0
    gt = gt_normal * 2.0 - 1.0
    pred = F.normalize(pred, p=2, dim=0)
    gt = F.normalize(gt, p=2, dim=0)
    cos_sim = torch.clamp(torch.sum(pred * gt, dim=0), -1.0, 1.0)
    ang_error = torch.acos(cos_sim) * (180.0 / math.pi)
    return ang_error.mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# Global per-channel albedo alignment
# ──────────────────────────────────────────────────────────────────────────────

def compute_global_per_channel_scale(rendered_list, gt_list):
    """Compute a single [3] scale vector across all images.

    For each channel c:
        s_c = sum_i sum_pixels( rendered_c * gt_c ) /
              sum_i sum_pixels( rendered_c^2 )

    Args:
        rendered_list: list of [3,H,W] tensors (predicted albedo)
        gt_list:       list of [3,H,W] tensors (ground truth albedo)

    Returns:
        scale: [3] tensor of per-channel scale factors.
    """
    num = torch.zeros(3, device=rendered_list[0].device)
    den = torch.zeros(3, device=rendered_list[0].device)
    for r, g in zip(rendered_list, gt_list):
        # r, g: [3, H, W]
        for c in range(3):
            num[c] += (r[c] * g[c]).sum()
            den[c] += (r[c] * r[c]).sum()
    scale = num / den.clamp_min(1e-8)
    return scale


def apply_channel_scale(img, scale):
    """Apply per-channel scale to a [3,H,W] image, clamp to [0,1]."""
    # scale: [3]
    return (img * scale[:, None, None]).clamp(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    # ── Load model + scene ────────────────────────────────────────────────
    model_path = os.path.abspath(args.model_path)
    dataset_path = os.path.abspath(args.source_path)

    # GaussianModel loads files with relative paths (e.g. load/lights/…),
    # so we must chdir into GIR/ – same as render.py is invoked via
    # subprocess with cwd="GIR" in relight_all.py.
    # Make sure all user-facing paths are absolute first.
    args.model_path = model_path
    args.source_path = dataset_path
    os.chdir(GIR_DIR)

    # Environment texture needed by GaussianModel init (uses default HDR)
    env_texture = getattr(args, "environment_texture", os.path.join(GIR_DIR, "hdri", "flower_road_no_sun_2k.hdr"))
    env_texture = os.path.abspath(env_texture)

    gaussians = GaussianModel(args.sh_degree, environment_texture=env_texture)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.get_diffuse_occ()

    iteration = scene.loaded_iter
    print(f"Loaded model at iteration {iteration}")

    # Background
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Pipeline params – need an object with .debug, .compute_cov3D_python, etc.
    pipe = args  # get_combined_args already merged everything

    test_cameras = scene.getTestCameras()
    print(f"Evaluating {len(test_cameras)} test views")

    # Read second_stage_step from cfg_args (fallback to default 30000)
    second_stage_step = getattr(args, "second_stage_step", 30000)
    first_stage_step = getattr(args, "first_stage_step", 5000)
    hdr_rotation = getattr(args, "hdr_rotation", True)
    remove_noise = getattr(args, "remove_noise", False)
    disable_sh = getattr(args, "disable_sh", False)

    # We force iteration to be > second_stage_step so the renderer produces
    # the full PBR decomposition (albedo, normal, material maps).
    render_iteration = max(iteration, second_stage_step + 1)

    # ── Pass 1: Render all views, collect albedo + normal data ────────────
    albedo_dir = os.path.join(model_path, "albedo_evaluation_aftermath")
    normal_dir = os.path.join(model_path, "normal_evaluation_aftermath")
    os.makedirs(albedo_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    rendered_albedos = []
    gt_albedos = []
    rendered_normals = []
    gt_normals = []
    view_names = []

    has_albedo_gt = False
    has_normal_gt = False

    print("\n── Pass 1: Rendering all test views ──")
    for idx, viewpoint in enumerate(tqdm(test_cameras, desc="Rendering")):
        with torch.no_grad():
            render_pkg = render(
                viewpoint, gaussians, pipe, background,
                iteration=render_iteration, is_train=False,
                first_stage_step=first_stage_step,
                second_stage_step=second_stage_step,
                remove_noise=remove_noise,
                hdr_rotation=hdr_rotation,
                disable_sh=disable_sh,
            )

        view_name = getattr(viewpoint, "image_name", f"{idx:05d}")
        view_names.append(view_name)

        # ── Albedo ────────────────────────────────────────────────────────
        rendered_albedo = render_pkg.get("rendered_albedo", None)
        gt_albedo = getattr(viewpoint, "albedo_gt", None)

        if rendered_albedo is not None:
            rendered_albedo = rendered_albedo.clamp(0.0, 1.0)
            torchvision.utils.save_image(
                rendered_albedo,
                os.path.join(albedo_dir, f"{view_name}_rendered_albedo.png"),
            )

        if gt_albedo is not None:
            gt_albedo = gt_albedo.to("cuda").clamp(0.0, 1.0)
            torchvision.utils.save_image(
                gt_albedo,
                os.path.join(albedo_dir, f"{view_name}_gt_albedo.png"),
            )

        if rendered_albedo is not None and gt_albedo is not None:
            has_albedo_gt = True
            rendered_albedos.append(rendered_albedo)
            gt_albedos.append(gt_albedo)

        # ── Normal ────────────────────────────────────────────────────────
        rendered_normal = render_pkg.get("rendered_normal", None)
        gt_normal = getattr(viewpoint, "normal_gt", None)

        if rendered_normal is not None:
            torchvision.utils.save_image(
                rendered_normal,
                os.path.join(normal_dir, f"{view_name}_rendered_normal.png"),
            )

        if gt_normal is not None:
            gt_normal = gt_normal.to("cuda")
            torchvision.utils.save_image(
                gt_normal,
                os.path.join(normal_dir, f"{view_name}_gt_normal.png"),
            )

        if rendered_normal is not None and gt_normal is not None:
            has_normal_gt = True
            rendered_normals.append(rendered_normal)
            gt_normals.append(gt_normal)

    # ── Pass 2: Compute albedo metrics ────────────────────────────────────
    metrics = {"iteration": iteration, "num_test_views": len(test_cameras)}
    albedo_metrics = {}
    normal_metrics = {}

    if has_albedo_gt:
        print("\n── Pass 2: Computing albedo metrics ──")

        # Global per-channel scale (closed-form least squares)
        scale = compute_global_per_channel_scale(rendered_albedos, gt_albedos)
        print(f"Global per-channel scale: R={scale[0]:.4f}, G={scale[1]:.4f}, B={scale[2]:.4f}")

        raw_psnrs = []
        raw_ssims = []
        aligned_psnrs = []
        per_view_albedo = []

        for i, (rend, gt) in enumerate(zip(rendered_albedos, gt_albedos)):
            # Raw metrics
            raw_psnrs.append(psnr_torch(rend, gt))
            raw_ssims.append(compute_ssim(rend, gt).item())

            # Aligned metrics
            aligned = apply_channel_scale(rend, scale)
            aligned_psnrs.append(psnr_torch(aligned, gt))

            # Save aligned albedo image
            torchvision.utils.save_image(
                aligned,
                os.path.join(albedo_dir, f"{view_names[i]}_aligned_albedo.png"),
            )

            # Save error map (abs diff of aligned, averaged over channels, heat-mapped)
            err = (aligned - gt).abs().mean(dim=0, keepdim=True).repeat(3, 1, 1)
            torchvision.utils.save_image(
                err,
                os.path.join(albedo_dir, f"{view_names[i]}_albedo_error.png"),
            )

            per_view_albedo.append({
                "view": view_names[i],
                "raw_psnr": raw_psnrs[-1],
                "raw_ssim": raw_ssims[-1],
                "aligned_psnr": aligned_psnrs[-1],
            })

        albedo_metrics = {
            "global_channel_scale": {"R": scale[0].item(), "G": scale[1].item(), "B": scale[2].item()},
            "mean_raw_psnr": float(np.mean(raw_psnrs)),
            "mean_raw_ssim": float(np.mean(raw_ssims)),
            "mean_aligned_psnr": float(np.mean(aligned_psnrs)),
            "per_view": per_view_albedo,
        }
        print(f"  Raw  PSNR : {albedo_metrics['mean_raw_psnr']:.2f} dB")
        print(f"  Raw  SSIM : {albedo_metrics['mean_raw_ssim']:.4f}")
        print(f"  Aligned PSNR (per-ch scale): {albedo_metrics['mean_aligned_psnr']:.2f} dB")
    else:
        print("\nNo albedo ground truth found – skipping albedo metrics.")

    # ── Pass 3: Compute normal metrics ────────────────────────────────────
    if has_normal_gt:
        print("\n── Pass 3: Computing normal metrics ──")
        angular_errors = []
        per_view_normal = []

        for i, (rend, gt) in enumerate(zip(rendered_normals, gt_normals)):
            ae = angular_error_torch(rend, gt)
            angular_errors.append(ae)

            # Save error map (angular error visualized as grayscale, normalised)
            pred = rend * 2.0 - 1.0
            gt_n = gt * 2.0 - 1.0
            pred = F.normalize(pred, p=2, dim=0)
            gt_n = F.normalize(gt_n, p=2, dim=0)
            cos_sim = torch.clamp(torch.sum(pred * gt_n, dim=0), -1.0, 1.0)
            ang_map = torch.acos(cos_sim) * (180.0 / math.pi)  # [H,W]
            # Normalise to [0,1] for visualisation (cap at 90°)
            ang_vis = (ang_map / 90.0).clamp(0.0, 1.0).unsqueeze(0).repeat(3, 1, 1)
            torchvision.utils.save_image(
                ang_vis,
                os.path.join(normal_dir, f"{view_names[i]}_normal_error.png"),
            )

            per_view_normal.append({
                "view": view_names[i],
                "angular_error_deg": ae,
            })

        normal_metrics = {
            "mean_angular_error_deg": float(np.mean(angular_errors)),
            "per_view": per_view_normal,
        }
        print(f"  Mean angular error: {normal_metrics['mean_angular_error_deg']:.2f}°")
    else:
        print("\nNo normal ground truth found – skipping normal metrics.")

    # ── Write JSON ────────────────────────────────────────────────────────
    metrics["albedo"] = albedo_metrics
    metrics["normal"] = normal_metrics

    json_path = os.path.join(model_path, "material_evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("MATERIAL EVALUATION SUMMARY")
    print("=" * 60)
    if albedo_metrics:
        print(f"  Albedo (raw  PSNR)          : {albedo_metrics['mean_raw_psnr']:.2f} dB")
        print(f"  Albedo (raw  SSIM)          : {albedo_metrics['mean_raw_ssim']:.4f}")
        print(f"  Albedo (aligned PSNR, per-ch): {albedo_metrics['mean_aligned_psnr']:.2f} dB")
        s = albedo_metrics["global_channel_scale"]
        print(f"  Scale vector [R,G,B]        : [{s['R']:.4f}, {s['G']:.4f}, {s['B']:.4f}]")
    if normal_metrics:
        print(f"  Normal angular error        : {normal_metrics['mean_angular_error_deg']:.2f}°")
    print("=" * 60)
    print(f"  Rendered images → {albedo_dir}")
    print(f"                    {normal_dir}")
    print(f"  Metrics JSON    → {json_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser(description="Post-hoc material (albedo + normal) evaluation for a trained PBR-3DGS model.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int,
                        help="Model iteration to load (-1 = highest available)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--environment_texture", type=str,
                        default=os.path.join(GIR_DIR, "hdri", "flower_road_no_sun_2k.hdr"),
                        help="Path to environment HDR (needed for GaussianModel init)")
    # Training hyper-params that affect rendering decomposition
    parser.add_argument("--second_stage_step", default=30000, type=int,
                        help="Second stage step from training (controls when PBR decomposition activates)")
    parser.add_argument("--first_stage_step", default=5000, type=int)
    parser.add_argument("--hdr_rotation", action="store_true")
    parser.add_argument("--remove_noise", action="store_true")
    parser.add_argument("--disable_sh", action="store_true")

    args = get_combined_args(parser)
    # Ensure we always evaluate against the actual ground truth, not training-time priors,
    # unless a custom directory was explicitly specified on the command line.
    cmdline_args = parser.parse_known_args()[0]
    if cmdline_args.albedo_gt_dir is None:
        args.albedo_gt_dir = "albedo_gt"
    if cmdline_args.normal_gt_dir is None:
        args.normal_gt_dir = "normal_gt"
    # Ensure eval mode so the Scene constructor keeps train/test split
    args.eval = True

    safe_state(getattr(args, "quiet", False))

    with torch.no_grad():
        evaluate(args)
