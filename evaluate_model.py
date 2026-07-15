#!/usr/bin/env python3
"""
Unified evaluation script for PBR-3DGS models.

Loads a trained model, renders test views for material decomposition and
relighting under all available HDRIs, and computes a comprehensive set of
metrics.  No prior evaluation scripts need to be run — this works on a
freshly finished training run.

Metrics produced:
  Albedo  → PSNR, SSIM, LPIPS  (after global per-channel [R,G,B] scale alignment)
  Normal  → Mean Angular Error (degrees)
  Relight → Mean PSNR, SSIM, LPIPS  (after per-image scalar scale alignment)
            averaged across all HDRIs that have ground truth.

Usage:
    python evaluate_model.py -m <model_path> [--hdris city courtyard snow]

Output JSON (one-row DataFrame-ready):
    {
      "model": "...",
      "albedo_psnr": ..., "albedo_ssim": ..., "albedo_lpips": ...,
      "normal_mae": ...,
      "relight_psnr": ..., "relight_ssim": ..., "relight_lpips": ...
    }

Python usage:
    import pandas as pd, json
    rows = [json.load(open(p)) for p in ["model_a/evaluation_metrics.json",
                                          "model_b/evaluation_metrics.json"]]
    df = pd.DataFrame(rows)
    print(df.to_string())
"""

import os
import sys
import json
import math
import glob
import argparse

import numpy as np

# ---------------------------------------------------------------------------
# GIR path setup — must happen before any GIR imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GIR_DIR = os.path.join(SCRIPT_DIR, "GIR")
sys.path.insert(0, GIR_DIR)

import torch
import torch.nn.functional as F
import torchvision

from argparse import Namespace
from tqdm import tqdm

from scene import Scene
from gaussian_renderer import render as gs_render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from utils.loss_utils import ssim as compute_ssim_torch
from lpipsPyTorch.modules.lpips import LPIPS as LPIPSModel


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def psnr_torch(img, gt):
    """PSNR between two [3,H,W] or [C,H,W] float tensors in [0,1]."""
    mse = torch.mean((img - gt) ** 2)
    if mse < 1e-10:
        return float("inf")
    return (-10.0 * torch.log10(mse)).item()


def angular_error_torch(pred, gt):
    """Mean angular error (degrees) between two [3,H,W] normal maps in [0,1]."""
    pred_n = pred * 2.0 - 1.0
    gt_n = gt * 2.0 - 1.0
    pred_n = F.normalize(pred_n, p=2, dim=0)
    gt_n = F.normalize(gt_n, p=2, dim=0)
    cos_sim = torch.clamp(torch.sum(pred_n * gt_n, dim=0), -1.0, 1.0)
    ang = torch.acos(cos_sim) * (180.0 / math.pi)
    return ang.mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# Scale alignment
# ──────────────────────────────────────────────────────────────────────────────

def global_per_channel_scale(rendered_list, gt_list):
    """Closed-form least-squares per-channel scale [3].

    For each channel c:  s_c = Σ(rendered_c · gt_c) / Σ(rendered_c²)
    Computed across ALL images simultaneously.
    """
    num = torch.zeros(3, device=rendered_list[0].device)
    den = torch.zeros(3, device=rendered_list[0].device)
    for r, g in zip(rendered_list, gt_list):
        for c in range(3):
            num[c] += (r[c] * g[c]).sum()
            den[c] += (r[c] * r[c]).sum()
    return num / den.clamp_min(1e-8)


def per_image_scalar_scale(render, gt):
    """Per-image scalar gain: s = Σ(render · gt) / Σ(render²)."""
    num = (render * gt).sum()
    den = (render * render).sum()
    return (num / den.clamp_min(1e-8)).item()


def get_image_name_with_underscores(viewpoint, dataset_path):
    image_path = getattr(viewpoint, "image_path", None)
    if image_path:
        abs_img_path = os.path.abspath(image_path)
        abs_dataset_path = os.path.abspath(dataset_path)
        rel_path = os.path.relpath(abs_img_path, abs_dataset_path)
        rel_path_no_ext = os.path.splitext(rel_path)[0]
        return rel_path_no_ext.replace("/", "_").replace("\\", "_").replace(".", "_")
    else:
        return viewpoint.image_name.replace("/", "_").replace("\\", "_").replace(".", "_")



# ──────────────────────────────────────────────────────────────────────────────
# Image I/O (for relighting GT from disk)
# ──────────────────────────────────────────────────────────────────────────────

def load_image_torch(path, device="cuda"):
    """Load a PNG as a float32 [3,H,W] tensor in [0,1]."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


def resize_to_match(img, target):
    """Resize img [3,H,W] to match target's spatial dims if needed."""
    if img.shape[1:] != target.shape[1:]:
        img = F.interpolate(
            img.unsqueeze(0), size=target.shape[1:],
            mode="bilinear", align_corners=False,
        ).squeeze(0)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(args, hdri_filter=None, use_lli=False):
    """Run the full evaluation pipeline.

    Args:
        args: Merged Namespace from get_combined_args (contains model_path,
              source_path, sh_degree, white_background, etc.)
        hdri_filter: Optional list of HDRI names to evaluate (default: all
                     that have GT in the dataset).
        use_lli: If True, enable light_linear_indirect for relighting renders
                 and store metrics under relight_lli_* keys.
    Returns:
        dict: Flat metrics dict suitable for pd.DataFrame.
    """
    model_path = os.path.abspath(args.model_path)
    dataset_path = os.path.abspath(args.source_path)

    # GaussianModel uses relative paths (e.g. load/lights/…) so we must
    # chdir into GIR/ — same as render.py and evaluate_materials.py.
    args.model_path = model_path
    args.source_path = dataset_path
    os.chdir(GIR_DIR)

    # ── Load model ────────────────────────────────────────────────────────
    env_texture = getattr(
        args, "environment_texture",
        os.path.join(GIR_DIR, "hdri", "flower_road_no_sun_2k.hdr"),
    )
    env_texture = os.path.abspath(env_texture)

    print("Loading model …")
    gaussians = GaussianModel(args.sh_degree, environment_texture=env_texture)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.get_diffuse_occ()

    iteration = scene.loaded_iter
    print(f"Loaded model at iteration {iteration}")

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = args  # merged namespace has all pipeline fields

    second_stage_step = getattr(args, "second_stage_step", 30000)
    first_stage_step = getattr(args, "first_stage_step", 5000)
    hdr_rotation = getattr(args, "hdr_rotation", True)
    remove_noise = getattr(args, "remove_noise", False)
    light_linear_indirect = getattr(args, "light_linear_indirect", False)

    # Override light_linear_indirect for relighting if --lli was requested
    relight_lli = use_lli

    # Force iteration past second_stage_step so PBR decomposition is active
    render_iteration = max(iteration, second_stage_step + 1)

    test_cameras = scene.getTestCameras()
    print(f"Evaluating {len(test_cameras)} test views")

    # ── Initialise LPIPS model ────────────────────────────────────────────
    print("Loading LPIPS (VGG) model …")
    lpips_model = LPIPSModel("vgg", "0.1").cuda()
    lpips_model.eval()

    # =====================================================================
    # PART 1: Material evaluation (Albedo + Normal)
    # =====================================================================
    print("\n" + "=" * 60)
    print("PART 1: Material Evaluation (Albedo + Normal)")
    print("=" * 60)

    rendered_albedos, gt_albedos = [], []
    albedo_cameras = []
    angular_errors = []

    for idx, viewpoint in enumerate(tqdm(test_cameras, desc="Rendering materials")):
        render_pkg = gs_render(
            viewpoint, gaussians, pipe, background,
            iteration=render_iteration, is_train=False,
            first_stage_step=first_stage_step,
            second_stage_step=second_stage_step,
            remove_noise=remove_noise,
            hdr_rotation=hdr_rotation,
            light_linear_indirect=light_linear_indirect,
        )

        # ── Albedo ────────────────────────────────────────────────────
        rendered_albedo = render_pkg.get("rendered_albedo", None)
        gt_albedo = getattr(viewpoint, "albedo_gt", None)

        if rendered_albedo is not None and gt_albedo is not None:
            rendered_albedo = rendered_albedo.clamp(0.0, 1.0)
            gt_albedo = gt_albedo.to("cuda").clamp(0.0, 1.0)
            rendered_albedos.append(rendered_albedo)
            gt_albedos.append(gt_albedo)
            albedo_cameras.append(viewpoint)

        # ── Normal ────────────────────────────────────────────────────
        rendered_normal = render_pkg.get("rendered_normal", None)
        gt_normal = getattr(viewpoint, "normal_gt", None)

        if rendered_normal is not None and gt_normal is not None:
            gt_normal = gt_normal.to("cuda")
            angular_errors.append(angular_error_torch(rendered_normal, gt_normal))

    # ── Albedo metrics ────────────────────────────────────────────────────
    row = {"model": os.path.basename(model_path)}

    if rendered_albedos:
        scale = global_per_channel_scale(rendered_albedos, gt_albedos)
        print(f"\n  Albedo global per-channel scale: "
              f"R={scale[0]:.4f}, G={scale[1]:.4f}, B={scale[2]:.4f}")

        # Create output directory for albedo renders
        albedo_dir_out = os.path.join(model_path, "evaluation_albedo")
        os.makedirs(albedo_dir_out, exist_ok=True)

        a_psnrs, a_ssims, a_lpipss = [], [], []
        a_psnrs_raw, a_ssims_raw, a_lpipss_raw = [], [], []
        for r, g, cam in zip(rendered_albedos, gt_albedos, albedo_cameras):
            # Raw (unaligned)
            a_psnrs_raw.append(psnr_torch(r, g))
            a_ssims_raw.append(compute_ssim_torch(r, g).item())
            a_lpipss_raw.append(lpips_model(r.unsqueeze(0), g.unsqueeze(0)).item())
            # Scale-aligned
            aligned = (r * scale[:, None, None]).clamp(0.0, 1.0)
            a_psnrs.append(psnr_torch(aligned, g))
            a_ssims.append(compute_ssim_torch(aligned, g).item())
            a_lpipss.append(lpips_model(aligned.unsqueeze(0), g.unsqueeze(0)).item())

            # Save visual comparisons
            img_name_with_underscores = get_image_name_with_underscores(cam, dataset_path)
            torchvision.utils.save_image(g, os.path.join(albedo_dir_out, f"{img_name_with_underscores}_gt.png"))
            torchvision.utils.save_image(aligned, os.path.join(albedo_dir_out, f"{img_name_with_underscores}_rendered.png"))

        row["albedo_psnr_raw"] = float(np.mean(a_psnrs_raw))
        row["albedo_ssim_raw"] = float(np.mean(a_ssims_raw))
        row["albedo_lpips_raw"] = float(np.mean(a_lpipss_raw))
        row["albedo_psnr"] = float(np.mean(a_psnrs))
        row["albedo_ssim"] = float(np.mean(a_ssims))
        row["albedo_lpips"] = float(np.mean(a_lpipss))
        print(f"  Albedo (raw)     → PSNR={row['albedo_psnr_raw']:.2f}  "
              f"SSIM={row['albedo_ssim_raw']:.4f}  LPIPS={row['albedo_lpips_raw']:.4f}")
        print(f"  Albedo (aligned) → PSNR={row['albedo_psnr']:.2f}  "
              f"SSIM={row['albedo_ssim']:.4f}  LPIPS={row['albedo_lpips']:.4f}")
    else:
        print("\n  No albedo GT found — skipping albedo metrics.")

    # ── Normal metrics ────────────────────────────────────────────────────
    if angular_errors:
        row["normal_mae"] = float(np.mean(angular_errors))
        print(f"  Normal → MAE = {row['normal_mae']:.2f}°")
    else:
        print("\n  No normal GT found — skipping normal metrics.")

    # =====================================================================
    # PART 2: Relighting evaluation
    # =====================================================================
    lli_tag = " (with LLI)" if relight_lli else ""
    metric_prefix = "relight_lli" if relight_lli else "relight"

    print("\n" + "=" * 60)
    print(f"PART 2: Relighting Evaluation{lli_tag}")
    print("=" * 60)

    # Find HDRIs directory
    hdris_dir = os.path.join(dataset_path, "hdris")
    if not os.path.isdir(hdris_dir):
        print(f"  No HDRIs directory found at {hdris_dir} — skipping relighting.")
    else:
        # Discover HDRIs that have GT in the dataset
        all_hdri_files = sorted(glob.glob(os.path.join(hdris_dir, "*.hdr")))
        relight_psnrs, relight_ssims, relight_lpipss = [], [], []
        relight_psnrs_raw, relight_ssims_raw, relight_lpipss_raw = [], [], []
        hdris_evaluated = 0

        for hdri_path in all_hdri_files:
            hdri_name = os.path.splitext(os.path.basename(hdri_path))[0]

            # Apply HDRI filter if specified
            if hdri_filter is not None and hdri_name not in hdri_filter:
                continue

            # Check for GT: test/rgba_<hdri_name>/ in the dataset
            gt_dir = os.path.join(dataset_path, "test", f"rgba_{hdri_name}")
            if not os.path.isdir(gt_dir):
                gt_dir = os.path.join(dataset_path, f"rgba_{hdri_name}")
            if not os.path.isdir(gt_dir):
                continue

            gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
            if not gt_files:
                continue

            print(f"\n  Relighting: {hdri_name} ({len(gt_files)} GT views) …")

            # Swap environment map on the model
            gaussians.envlight.load(hdri_path)

            # Create output directory for relighting renders
            relight_dir_out = os.path.join(model_path, f"evaluation_{hdri_name}")
            os.makedirs(relight_dir_out, exist_ok=True)

            # Render all test views under this HDRI
            hdri_psnrs, hdri_ssims, hdri_lpipss = [], [], []
            hdri_psnrs_raw, hdri_ssims_raw, hdri_lpipss_raw = [], [], []
            hdri_gains = []
            num_views = min(len(test_cameras), len(gt_files))

            for idx in tqdm(range(num_views), desc=f"  {hdri_name}", leave=False):
                viewpoint = test_cameras[idx]
                render_pkg = gs_render(
                    viewpoint, gaussians, pipe, background,
                    iteration=render_iteration, is_train=False,
                    first_stage_step=first_stage_step,
                    second_stage_step=second_stage_step,
                    remove_noise=remove_noise,
                    hdr_rotation=hdr_rotation,
                    light_linear_indirect=relight_lli if relight_lli else light_linear_indirect,
                )
                rendered = render_pkg["render"].clamp(0.0, 1.0)

                # Load GT
                gt = load_image_torch(gt_files[idx], device="cuda")
                gt = resize_to_match(gt, rendered)

                # Raw (unaligned) metrics
                hdri_psnrs_raw.append(psnr_torch(rendered, gt))
                hdri_ssims_raw.append(compute_ssim_torch(rendered, gt).item())
                hdri_lpipss_raw.append(
                    lpips_model(rendered.unsqueeze(0), gt.unsqueeze(0)).item()
                )

                # Per-image scalar scale alignment
                gain = per_image_scalar_scale(rendered, gt)
                hdri_gains.append(gain)
                aligned = (rendered * gain).clamp(0.0, 1.0)

                hdri_psnrs.append(psnr_torch(aligned, gt))
                hdri_ssims.append(compute_ssim_torch(aligned, gt).item())
                hdri_lpipss.append(
                    lpips_model(aligned.unsqueeze(0), gt.unsqueeze(0)).item()
                )

                # Save visual comparisons
                img_name_with_underscores = get_image_name_with_underscores(viewpoint, dataset_path)
                torchvision.utils.save_image(gt, os.path.join(relight_dir_out, f"{img_name_with_underscores}_gt.png"))
                torchvision.utils.save_image(aligned, os.path.join(relight_dir_out, f"{img_name_with_underscores}_rendered.png"))

            if hdri_psnrs:
                hdris_evaluated += 1
                relight_psnrs.extend(hdri_psnrs)
                relight_ssims.extend(hdri_ssims)
                relight_lpipss.extend(hdri_lpipss)
                relight_psnrs_raw.extend(hdri_psnrs_raw)
                relight_ssims_raw.extend(hdri_ssims_raw)
                relight_lpipss_raw.extend(hdri_lpipss_raw)

                # Store per-HDRI metrics
                row[f"{metric_prefix}_{hdri_name}_psnr_raw"] = float(np.mean(hdri_psnrs_raw))
                row[f"{metric_prefix}_{hdri_name}_ssim_raw"] = float(np.mean(hdri_ssims_raw))
                row[f"{metric_prefix}_{hdri_name}_lpips_raw"] = float(np.mean(hdri_lpipss_raw))
                row[f"{metric_prefix}_{hdri_name}_psnr"] = float(np.mean(hdri_psnrs))
                row[f"{metric_prefix}_{hdri_name}_ssim"] = float(np.mean(hdri_ssims))
                row[f"{metric_prefix}_{hdri_name}_lpips"] = float(np.mean(hdri_lpipss))

                print(f"    {hdri_name} (raw):     PSNR={np.mean(hdri_psnrs_raw):.2f}  "
                      f"SSIM={np.mean(hdri_ssims_raw):.4f}  "
                      f"LPIPS={np.mean(hdri_lpipss_raw):.4f}")
                print(f"    {hdri_name} (aligned): PSNR={np.mean(hdri_psnrs):.2f}  "
                      f"SSIM={np.mean(hdri_ssims):.4f}  "
                      f"LPIPS={np.mean(hdri_lpipss):.4f}  "
                      f"(mean gain={np.mean(hdri_gains):.3f})")

        if relight_psnrs:
            # Overall average across all HDRIs
            row[f"{metric_prefix}_avg_psnr_raw"] = float(np.mean(relight_psnrs_raw))
            row[f"{metric_prefix}_avg_ssim_raw"] = float(np.mean(relight_ssims_raw))
            row[f"{metric_prefix}_avg_lpips_raw"] = float(np.mean(relight_lpipss_raw))
            row[f"{metric_prefix}_avg_psnr"] = float(np.mean(relight_psnrs))
            row[f"{metric_prefix}_avg_ssim"] = float(np.mean(relight_ssims))
            row[f"{metric_prefix}_avg_lpips"] = float(np.mean(relight_lpipss))
            print(f"\n  Relight{lli_tag} AVERAGE (over {hdris_evaluated} HDRIs, "
                  f"{len(relight_psnrs)} total views) →")
            print(f"    (raw)     PSNR={row[f'{metric_prefix}_avg_psnr_raw']:.2f}  "
                  f"SSIM={row[f'{metric_prefix}_avg_ssim_raw']:.4f}  "
                  f"LPIPS={row[f'{metric_prefix}_avg_lpips_raw']:.4f}")
            print(f"    (aligned) PSNR={row[f'{metric_prefix}_avg_psnr']:.2f}  "
                  f"SSIM={row[f'{metric_prefix}_avg_ssim']:.4f}  "
                  f"LPIPS={row[f'{metric_prefix}_avg_lpips']:.4f}")
        else:
            print("\n  No relighting GT matched any available HDRI.")

    return row


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified PBR-3DGS evaluation: Albedo, Normal, Relighting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output format (JSON, one row per model — loadable as pd.DataFrame):
  {"model": "...", "albedo_psnr": ..., "albedo_ssim": ..., "albedo_lpips": ...,
   "normal_mae": ..., "relight_psnr": ..., "relight_ssim": ..., "relight_lpips": ...}
""",
    )
    # Re-use GIR's ModelParams / PipelineParams so that cfg_args is auto-loaded
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int,
                        help="Model iteration to load (-1 = highest available)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--environment_texture", type=str,
                        default=os.path.join(GIR_DIR, "hdri", "flower_road_no_sun_2k.hdr"),
                        help="Default environment HDR for model init")
    parser.add_argument("--second_stage_step", default=30000, type=int)
    parser.add_argument("--first_stage_step", default=5000, type=int)
    parser.add_argument("--hdr_rotation", action="store_true")
    parser.add_argument("--remove_noise", action="store_true")
    parser.add_argument("--light_linear_indirect", action="store_true")
    parser.add_argument("--lli", action="store_true",
                        help="Enable light_linear_indirect for relighting evaluation "
                             "(metrics stored under relight_lli_* keys)")
    parser.add_argument("--hdris", nargs="+", default=None, type=str,
                        help="Only evaluate these HDRIs (names without .hdr)")
    parser.add_argument("-o", "--output", default=None, type=str,
                        help="Output JSON path (default: <model_path>/evaluation_metrics.json)")

    args = get_combined_args(parser)

    # Force GT directories for evaluation (not training priors)
    cmdline_args = parser.parse_known_args()[0]
    if cmdline_args.albedo_gt_dir is None:
        args.albedo_gt_dir = "albedo_gt"
    if cmdline_args.normal_gt_dir is None:
        args.normal_gt_dir = "normal_gt"
    args.eval = True

    safe_state(getattr(args, "quiet", False))

    model_path = os.path.abspath(args.model_path)
    output_path = getattr(cmdline_args, "output", None) or os.path.join(
        model_path, "evaluation_metrics.json"
    )
    hdri_filter = getattr(cmdline_args, "hdris", None)
    use_lli = getattr(cmdline_args, "lli", False)

    print("=" * 70)
    print("PBR-3DGS Unified Evaluation")
    print(f"  Model:   {model_path}")
    print(f"  Dataset: {os.path.abspath(args.source_path)}")
    if hdri_filter:
        print(f"  HDRIs:   {hdri_filter}")
    if use_lli:
        print(f"  LLI:     enabled (light_linear_indirect for relighting)")
    print("=" * 70)

    with torch.no_grad():
        row = evaluate(args, hdri_filter=hdri_filter, use_lli=use_lli)

    # Save JSON
    with open(output_path, "w") as f:
        json.dump(row, f, indent=4)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Value':>12}")
    print("  " + "-" * 40)
    for k, v in row.items():
        if k == "model":
            print(f"  {'Model':<25} {v:>12}")
        elif v is not None:
            print(f"  {k:<25} {v:>12.4f}")
    print("=" * 70)
    print(f"Saved to: {output_path}")

    print(f"\n# Python copy-paste:")
    print(f"row = {json.dumps(row, indent=4)}")
