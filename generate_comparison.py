"""
Generate a PDF comparison report for GIR render outputs.
Usage: python generate_comparison.py

Requires: pip install matplotlib  (if not already installed)
"""

import os
import sys
import glob
import numpy as np
from PIL import Image

# Try importing matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError:
    print("ERROR: matplotlib not found. Install it with:")
    print("  pip install matplotlib")
    sys.exit(1)

# Try importing torch + metrics from the project
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GIR'))
    import torch
    import torchvision.transforms.functional as TF
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ─── Configuration ───────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
ITERATION = 15000
SECOND_STAGE_STEP = 15000

SETS = {
    "Render (vs GT)":   os.path.join(OUTPUT_DIR, "test_test_render",    f"ours_{ITERATION}", "renders"),
    "Ground Truth":     os.path.join(OUTPUT_DIR, "test_test_render",    f"ours_{ITERATION}", "gt"),
    "Relight: Flower":  os.path.join(OUTPUT_DIR, "test_relight_flower", f"ours_{ITERATION}", "renders"),
    "Relight: Sky Fire":os.path.join(OUTPUT_DIR, "test_sky_is_on_fire", f"ours_{ITERATION}", "renders"),
}

PDF_PATH = os.path.join(os.path.dirname(__file__), "comparison_report.pdf")

# Number of sample views to show in the grid (evenly spaced)
NUM_SAMPLES = 8


# ─── Metrics ─────────────────────────────────────────────────────────────────
def compute_psnr_np(img1, img2):
    """PSNR between two float32 images in [0,1]."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return -10.0 * np.log10(mse)


def compute_ssim_np(img1, img2, window_size=11):
    """Simplified SSIM (per-channel mean) between two float32 images [H,W,C] in [0,1]."""
    from scipy.ndimage import uniform_filter
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


def try_compute_ssim(img1, img2):
    """Try scipy SSIM, fall back to None."""
    try:
        return compute_ssim_np(img1, img2)
    except ImportError:
        return None


def compute_lpips_torch(img1_np, img2_np):
    """Try to compute LPIPS using the project's lpipsPyTorch."""
    if not HAS_TORCH:
        return None
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GIR', 'lpipsPyTorch'))
        from lpips import LPIPS
        lpips_fn = LPIPS(net='vgg').cuda()
        t1 = torch.from_numpy(img1_np).permute(2,0,1).unsqueeze(0).float().cuda()
        t2 = torch.from_numpy(img2_np).permute(2,0,1).unsqueeze(0).float().cuda()
        with torch.no_grad():
            val = lpips_fn(t1, t2).item()
        del lpips_fn
        torch.cuda.empty_cache()
        return val
    except Exception as e:
        print(f"  LPIPS unavailable: {e}")
        return None


# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_image(path):
    """Load image as float32 numpy array [H,W,C] in [0,1]."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def get_sorted_images(directory):
    """Return sorted list of png paths."""
    return sorted(glob.glob(os.path.join(directory, "*.png")))


def pick_indices(total, n):
    """Pick n evenly spaced indices from [0, total)."""
    if total <= n:
        return list(range(total))
    return [int(i * (total - 1) / (n - 1)) for i in range(n)]


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print(f"Generating comparison report...")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Iteration:  {ITERATION}")

    # Validate directories
    for name, path in SETS.items():
        if not os.path.isdir(path):
            print(f"  WARNING: {name} directory not found: {path}")

    render_files = get_sorted_images(SETS["Render (vs GT)"])
    gt_files = get_sorted_images(SETS["Ground Truth"])
    flower_files = get_sorted_images(SETS["Relight: Flower"])
    fire_files = get_sorted_images(SETS["Relight: Sky Fire"])

    total = min(len(render_files), len(gt_files))
    if total == 0:
        print("ERROR: No images found!")
        sys.exit(1)

    print(f"  Total test views: {total}")

    # ── Compute metrics over ALL views ──
    print("\nComputing metrics over all views...")
    psnr_vals, ssim_vals, lpips_vals = [], [], []
    for i in range(total):
        gt = load_image(gt_files[i])
        rd = load_image(render_files[i])
        psnr_vals.append(compute_psnr_np(rd, gt))
        ssim_val = try_compute_ssim(rd, gt)
        if ssim_val is not None:
            ssim_vals.append(ssim_val)
        if i == 0:
            lp = compute_lpips_torch(rd, gt)
            if lp is not None:
                lpips_vals.append(lp)
                compute_all_lpips = True
            else:
                compute_all_lpips = False
        elif compute_all_lpips:
            lpips_vals.append(compute_lpips_torch(rd, gt))

        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{total}")

    avg_psnr = np.mean(psnr_vals)
    avg_ssim = np.mean(ssim_vals) if ssim_vals else None
    avg_lpips = np.mean(lpips_vals) if lpips_vals else None

    print(f"\n  Mean PSNR:  {avg_psnr:.2f} dB")
    if avg_ssim is not None:
        print(f"  Mean SSIM:  {avg_ssim:.4f}")
    if avg_lpips is not None:
        print(f"  Mean LPIPS: {avg_lpips:.4f}")

    # ── Pick sample views ──
    indices = pick_indices(total, NUM_SAMPLES)

    # ── Generate PDF ──
    print(f"\nGenerating PDF: {PDF_PATH}")
    with PdfPages(PDF_PATH) as pdf:
        # ─── Page 1: Title + Metrics ───
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.85, "GIR: PBR-3DGS Render Comparison",
                fontsize=24, fontweight='bold', ha='center', va='top',
                transform=ax.transAxes)
        ax.text(0.5, 0.75, f"Scene: NeRF Synthetic Lego",
                fontsize=16, ha='center', va='top', transform=ax.transAxes)
        ax.text(0.5, 0.68, f"Iteration: {ITERATION}  |  Second Stage Step: {SECOND_STAGE_STEP}",
                fontsize=14, ha='center', va='top', color='#555555',
                transform=ax.transAxes)

        # Metrics table
        metrics_text = f"{'Metric':<12} {'Value':>10}\n" + "─" * 24 + "\n"
        metrics_text += f"{'PSNR':<12} {avg_psnr:>9.2f} dB\n"
        if avg_ssim is not None:
            metrics_text += f"{'SSIM':<12} {avg_ssim:>10.4f}\n"
        if avg_lpips is not None:
            metrics_text += f"{'LPIPS':<12} {avg_lpips:>10.4f}\n"
        metrics_text += "─" * 24 + "\n"
        metrics_text += f"{'Test Views':<12} {total:>10d}\n"

        ax.text(0.5, 0.52, metrics_text,
                fontsize=14, ha='center', va='top', transform=ax.transAxes,
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f0f0', edgecolor='#cccccc'))

        ax.text(0.5, 0.18, "Comparison Sets:",
                fontsize=13, fontweight='bold', ha='center', va='top',
                transform=ax.transAxes)
        sets_text = "• Render vs Ground Truth (with PSNR/SSIM/LPIPS)\n"
        sets_text += "• Relighting: flower_road_no_sun_2k\n"
        sets_text += "• Relighting: the_sky_is_on_fire_2k"
        ax.text(0.5, 0.13, sets_text,
                fontsize=12, ha='center', va='top', transform=ax.transAxes,
                color='#333333')

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ─── Page 2: Render vs GT grid ───
        fig, axes = plt.subplots(2, NUM_SAMPLES, figsize=(NUM_SAMPLES * 2.5 + 1.5, 6))
        fig.suptitle(f"Render vs Ground Truth  (Iteration {ITERATION})", fontsize=16, fontweight='bold', y=0.98)

        row_labels_p2 = ["Ground Truth", "GIR Render"]
        row_colors_p2 = ["#2196F3", "#4CAF50"]
        for col, idx in enumerate(indices):
            gt_img = load_image(gt_files[idx])
            rd_img = load_image(render_files[idx])
            p = compute_psnr_np(rd_img, gt_img)

            axes[0, col].imshow(gt_img)
            axes[0, col].set_title(f"View {idx}", fontsize=8)
            axes[0, col].axis('off')

            axes[1, col].imshow(rd_img)
            axes[1, col].set_title(f"PSNR: {p:.1f} dB", fontsize=8, color='#006600')
            axes[1, col].axis('off')

        plt.tight_layout(rect=[0.08, 0, 1, 0.95])
        for row in range(2):
            bbox = axes[row, 0].get_position()
            y_center = (bbox.y0 + bbox.y1) / 2
            fig.text(0.04, y_center, row_labels_p2[row], fontsize=11, fontweight='bold',
                     ha='center', va='center', rotation=90, color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=row_colors_p2[row], alpha=0.9))
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ─── Page 3: Error map ───
        fig, axes = plt.subplots(2, NUM_SAMPLES, figsize=(NUM_SAMPLES * 2.5 + 1.5, 6))
        fig.suptitle("Absolute Error Maps  (x5 amplified)", fontsize=16, fontweight='bold', y=0.98)

        row_labels_p3 = ["GIR Render", "Error (x5)"]
        row_colors_p3 = ["#4CAF50", "#F44336"]
        for col, idx in enumerate(indices):
            gt_img = load_image(gt_files[idx])
            rd_img = load_image(render_files[idx])
            err = np.abs(rd_img - gt_img)
            err_amplified = np.clip(err * 5, 0, 1)

            axes[0, col].imshow(rd_img)
            axes[0, col].set_title(f"View {idx}", fontsize=8)
            axes[0, col].axis('off')

            axes[1, col].imshow(err_amplified)
            axes[1, col].axis('off')

        plt.tight_layout(rect=[0.08, 0, 1, 0.95])
        for row in range(2):
            bbox = axes[row, 0].get_position()
            y_center = (bbox.y0 + bbox.y1) / 2
            fig.text(0.04, y_center, row_labels_p3[row], fontsize=11, fontweight='bold',
                     ha='center', va='center', rotation=90, color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=row_colors_p3[row], alpha=0.9))
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ─── Page 4: Relighting comparison ───
        if flower_files and fire_files:
            fig, axes = plt.subplots(3, NUM_SAMPLES, figsize=(NUM_SAMPLES * 2.5 + 1.5, 8.5))
            fig.suptitle(f"Relighting Comparison  (Iteration {ITERATION})", fontsize=16, fontweight='bold', y=0.98)

            row_labels = ["Original Render", "Relight:\nFlower Road", "Relight:\nSky is on Fire"]
            row_colors = ["#4CAF50", "#FF9800", "#E91E63"]
            file_sets = [render_files, flower_files, fire_files]

            for row in range(3):
                for col, idx in enumerate(indices):
                    if idx < len(file_sets[row]):
                        img = load_image(file_sets[row][idx])
                        axes[row, col].imshow(img)
                    axes[row, col].axis('off')
                    if row == 0:
                        axes[row, col].set_title(f"View {idx}", fontsize=8)

            plt.tight_layout(rect=[0.09, 0, 1, 0.95])
            for row in range(3):
                bbox = axes[row, 0].get_position()
                y_center = (bbox.y0 + bbox.y1) / 2
                fig.text(0.045, y_center, row_labels[row], fontsize=10, fontweight='bold',
                         ha='center', va='center', rotation=90, color='white',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=row_colors[row], alpha=0.9))
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        # ─── Page 5: Per-view PSNR chart ───
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(range(total), psnr_vals, color='#4287f5', alpha=0.8, width=1.0)
        ax.axhline(y=avg_psnr, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {avg_psnr:.2f} dB')
        ax.set_xlabel("View Index", fontsize=12)
        ax.set_ylabel("PSNR (dB)", fontsize=12)
        ax.set_title(f"Per-View PSNR  (Iteration {ITERATION})", fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"\n✓ Report saved to: {PDF_PATH}")


if __name__ == "__main__":
    main()
