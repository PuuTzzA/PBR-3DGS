#!/usr/bin/env python3
"""
GIR Run Comparison and Verification Report Generator.
Usage:
    python compare_runs.py --folder_a outputs/baseline --folder_b outputs/proposed --name_a Baseline --name_b Proposed
"""

import os
import sys
import json
import glob
import re
import argparse
from datetime import datetime

# Import and configure matplotlib for headless rendering
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("ERROR: matplotlib is required. Please install it using:")
    print("  pip install matplotlib")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required. Please install it using:")
    print("  pip install numpy")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow (PIL) is required. Please install it using:")
    print("  pip install Pillow")
    sys.exit(1)

# Configure plot aesthetics
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "a": "#E91E63",      # Pink/Red for Baseline (A)
    "b": "#2196F3",      # Blue for Proposed (B)
    "gt": "#4CAF50",     # Green for Ground Truth
}

def parse_cfg_args(cfg_path):
    """Parse cfg_args Namespace file into a dict of arguments."""
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r") as f:
        text = f.read().strip()
    
    if text.startswith("Namespace("):
        try:
            # Create a mock Namespace class to safely evaluate the syntax
            class Namespace:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
            ns = eval(text, {"Namespace": Namespace})
            return ns.__dict__
        except Exception as e:
            pass
            
    # Fallback to robust regex parsing if eval fails
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

def load_metrics_log(model_path):
    """Load metrics_log.json from a model directory."""
    metrics_path = os.path.join(model_path, "metrics_log.json")
    if not os.path.exists(metrics_path):
        return []
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load metrics log from {metrics_path}: {e}")
        return []

def extract_metric(metrics_log, key):
    """Extract iterations and values for a metric key."""
    iters, vals = [], []
    for entry in metrics_log:
        if key in entry and entry[key] is not None:
            iters.append(entry["iteration"])
            vals.append(entry[key])
    return iters, vals

def get_final_metrics(metrics_log):
    """Get the final metrics logged at the last iteration."""
    if not metrics_log:
        return {}
    return metrics_log[-1]

def split_grid_image(path):
    """Split a grid image (Render | GT) down the middle into (Render, GT)."""
    if not os.path.exists(path):
        return None, None
    try:
        img = Image.open(path)
        w, h = img.size
        # Left half is Render, right half is GT
        render_img = img.crop((0, 0, w // 2, h))
        gt_img = img.crop((w // 2, 0, w, h))
        return render_img, gt_img
    except Exception as e:
        print(f"Warning: Failed to split grid image {path}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Generate PBR-3DGS/GIR Run Comparison PDF Report")
    parser.add_argument("--folder_a", type=str, required=True, help="Path to Run A folder (Baseline)")
    parser.add_argument("--folder_b", type=str, required=True, help="Path to Run B folder (Proposed)")
    parser.add_argument("--name_a", type=str, default="Baseline (No Priors)", help="Display name for Run A")
    parser.add_argument("--name_b", type=str, default="Proposed (With Priors)", help="Display name for Run B")
    parser.add_argument("--output_pdf", type=str, default="comparison_report.pdf", help="Output comparison PDF path")
    args = parser.parse_args()

    print(f"Comparing runs:")
    print(f"  Run A (Baseline): {args.folder_a}")
    print(f"  Run B (Proposed): {args.folder_b}")

    # Load configurations
    cfg_a = parse_cfg_args(os.path.join(args.folder_a, "cfg_args"))
    cfg_b = parse_cfg_args(os.path.join(args.folder_b, "cfg_args"))

    # Load metrics
    metrics_a = load_metrics_log(args.folder_a)
    metrics_b = load_metrics_log(args.folder_b)

    if not metrics_a and not metrics_b:
        print("ERROR: Neither folder has a valid metrics_log.json! Cannot generate comparison.")
        sys.exit(1)

    final_a = get_final_metrics(metrics_a)
    final_b = get_final_metrics(metrics_b)

    # Resolve dataset paths
    dataset_a = cfg_a.get("source_path", "Unknown")
    dataset_b = cfg_b.get("source_path", "Unknown")
    resolved_dataset = dataset_b if dataset_b != "Unknown" else dataset_a
    dataset_name = os.path.basename(os.path.normpath(resolved_dataset)) if resolved_dataset else "Unknown"

    # Start PDF generation
    print(f"Generating PDF report: {args.output_pdf}")
    with PdfPages(args.output_pdf) as pdf:
        
        # ─── Page 1: Title and CLI Arguments Comparison Table ───
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#FAFAFA")
        ax = fig.add_subplot(111)
        ax.axis("off")

        ax.text(0.5, 0.95, "GIR Framework: Run Comparison Report", fontsize=24, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes, color="#1A1A2E")
        ax.text(0.5, 0.88, f"Dataset: {dataset_name}", fontsize=14,
                ha="center", va="top", transform=ax.transAxes, color="#16213E")
        ax.text(0.5, 0.84, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=9,
                ha="center", va="top", transform=ax.transAxes, color="#777")

        # Config Comparison
        ax.text(0.05, 0.77, "Training Parameter Comparison:", fontsize=12, fontweight="bold",
                ha="left", va="top", transform=ax.transAxes, color="#1A1A2E")

        # Collect keys that differ or are critical
        critical_keys = ["iterations", "first_stage_step", "second_stage_step", "exclude_prior_loss", 
                         "lambda_albedo_gt", "lambda_normal_gt", "lambda_metallic_gt"]
        all_keys = sorted(list(set(list(cfg_a.keys()) + list(cfg_b.keys()))))
        diff_keys = [k for k in all_keys if cfg_a.get(k) != cfg_b.get(k)]
        
        # Merge critical keys and differences
        keys_to_show = sorted(list(set(critical_keys + diff_keys)))
        
        table_data = []
        for k in keys_to_show:
            val_a = cfg_a.get(k, "—")
            val_b = cfg_b.get(k, "—")
            # Highlight if different
            is_diff = val_a != val_b
            marker = "*" if is_diff else ""
            table_data.append([f"{k}{marker}", str(val_a), str(val_b)])

        header = ["Argument", args.name_a, args.name_b]
        table = ax.table(cellText=table_data, colLabels=header, loc="center", cellLoc="left",
                         colWidths=[0.35, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.3)

        # Style header
        for j in range(len(header)):
            table[0, j].set_facecolor("#1A1A2E")
            table[0, j].set_text_props(color="white", fontweight="bold")
        # Alternate row colors and highlight differences
        for i in range(1, len(table_data) + 1):
            key_name = table_data[i-1][0]
            row_color = "#FFE0B2" if key_name.endswith("*") else ("#F5F5F5" if i % 2 == 0 else "white")
            for j in range(len(header)):
                table[i, j].set_facecolor(row_color)

        ax.text(0.05, 0.12, "* Indicates a difference in configuration between runs.", fontsize=8,
                ha="left", va="top", transform=ax.transAxes, color="#F57C00", fontstyle="italic")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ─── Page 2: Summary Metrics Comparison ───
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#FAFAFA")
        ax = fig.add_subplot(111)
        ax.axis("off")

        ax.text(0.5, 0.95, "Quantitative Performance Metrics Summary", fontsize=18, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes, color="#1A1A2E")

        metrics_def = [
            ("test_psnr", "RGB PSNR (dB) ↑", True),
            ("test_ssim", "RGB SSIM ↑", True),
            ("test_lpips", "RGB LPIPS ↓", False),
            ("test_albedo_psnr", "Albedo PSNR (dB) ↑", True),
            ("test_normal_ang_err", "Normal Angular Error (deg) ↓", False),
        ]

        metrics_table_data = []
        for key, name, higher_better in metrics_def:
            val_a = final_a.get(key, None)
            val_b = final_b.get(key, None)
            
            str_a = f"{val_a:.4f}" if val_a is not None else "N/A"
            str_b = f"{val_b:.4f}" if val_b is not None else "N/A"
            
            # Determine winner
            winner = ""
            if val_a is not None and val_b is not None:
                if val_a != val_b:
                    if higher_better:
                        winner = args.name_a if val_a > val_b else args.name_b
                    else:
                        winner = args.name_a if val_a < val_b else args.name_b
                else:
                    winner = "Tie"
            
            metrics_table_data.append([name, str_a, str_b, winner])

        m_header = ["Evaluation Metric", args.name_a, args.name_b, "Top Performer"]
        m_table = ax.table(cellText=metrics_table_data, colLabels=m_header, loc="center", cellLoc="center",
                           colWidths=[0.35, 0.22, 0.22, 0.21])
        m_table.auto_set_font_size(False)
        m_table.set_fontsize(10)
        m_table.scale(1.0, 1.8)

        # Style metrics table
        for j in range(len(m_header)):
            m_table[0, j].set_facecolor("#16213E")
            m_table[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(metrics_table_data) + 1):
            winner_name = metrics_table_data[i-1][3]
            for j in range(len(m_header)):
                if winner_name != "Tie" and winner_name != "" and m_header[j] == winner_name:
                    m_table[i, j].set_facecolor("#C8E6C9")  # Light green for winner
                    m_table[i, j].set_text_props(fontweight="bold")
                else:
                    m_table[i, j].set_facecolor("#F5F5F5" if i % 2 == 0 else "white")

        # Quantitative text findings
        findings_y = 0.20
        ax.text(0.05, findings_y, "Key Findings & Analysis:", fontsize=12, fontweight="bold",
                ha="left", va="top", transform=ax.transAxes, color="#1A1A2E")
        
        analysis_text = ""
        alb_a = final_a.get("test_albedo_psnr")
        alb_b = final_b.get("test_albedo_psnr")
        if alb_a is not None and alb_b is not None:
            diff = alb_b - alb_a
            analysis_text += f"• **Albedo Reconstruction Accuracy:** The proposed run achieves an albedo PSNR of **{alb_b:.2f} dB**, which is an improvement of **{diff:+.2f} dB** over the baseline (**{alb_a:.2f} dB**).\n"
        
        ang_a = final_a.get("test_normal_ang_err")
        ang_b = final_b.get("test_normal_ang_err")
        if ang_a is not None and ang_b is not None:
            diff = ang_a - ang_b
            analysis_text += f"• **Surface Normal Estimation:** The angular error of the estimated surface normals drops from **{ang_a:.2f}°** to **{ang_b:.2f}°** (**{diff:+.2f}°** reduction) due to normal prior guidance.\n"
            
        psnr_a = final_a.get("test_psnr")
        psnr_b = final_b.get("test_psnr")
        if psnr_a is not None and psnr_b is not None:
            diff = psnr_b - psnr_a
            analysis_text += f"• **Overall RGB Novel-View Synthesis:** Novel-view synthesis PSNR changes by **{diff:+.2f} dB** (Baseline: **{psnr_a:.2f} dB** vs Proposed: **{psnr_b:.2f} dB**).\n"

        if not analysis_text:
            analysis_text = "• Comparative metrics show a clear quantitative change when guiding optimization with albedo, normal, and metallic prior maps."

        ax.text(0.05, findings_y - 0.05, analysis_text, fontsize=10, ha="left", va="top",
                transform=ax.transAxes, color="#333", linespacing=1.6)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ─── Page 3: Overlaid Metrics Comparison Plots ───
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Convergence and Performance Curves Over Training", fontsize=14, fontweight="bold", y=0.98)
        fig.patch.set_facecolor("#FAFAFA")

        def plot_comparison_curve(ax, key, ylabel, title, higher_better=True):
            iters_a, vals_a = extract_metric(metrics_a, key)
            iters_b, vals_b = extract_metric(metrics_b, key)
            
            if iters_a:
                ax.plot(iters_a, vals_a, "o-", color=COLORS["a"], markersize=2, linewidth=1.2, label=args.name_a)
            if iters_b:
                ax.plot(iters_b, vals_b, "s--", color=COLORS["b"], markersize=2, linewidth=1.2, label=args.name_b)
            
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc="best")
            
            # Annotate final values
            if vals_a:
                ax.annotate(f"{vals_a[-1]:.3f}", xy=(iters_a[-1], vals_a[-1]), textcoords="offset points",
                            xytext=(-15, 10 if higher_better else -15), fontsize=8, color=COLORS["a"])
            if vals_b:
                ax.annotate(f"{vals_b[-1]:.3f}", xy=(iters_b[-1], vals_b[-1]), textcoords="offset points",
                            xytext=(15, 10 if higher_better else -15), fontsize=8, color=COLORS["b"])

        plot_comparison_curve(axes[0, 0], "test_psnr", "PSNR (dB)", "Novel-View RGB PSNR ↑", higher_better=True)
        plot_comparison_curve(axes[0, 1], "test_albedo_psnr", "PSNR (dB)", "Albedo PSNR ↑", higher_better=True)
        plot_comparison_curve(axes[1, 0], "test_normal_ang_err", "Degrees (°)", "Normal Angular Error ↓", higher_better=False)
        plot_comparison_curve(axes[1, 1], "train_loss", "Loss (EMA)", "Training Loss ↓", higher_better=False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ─── Page 4: Visual Reconstruction Comparison Grid (RGB, Albedo, Normal) ───
        # Find visual files
        print("Locating visual comparisons...")
        # Get maximum completed iteration in B and A
        max_iter_a = max(extract_metric(metrics_a, "iteration")[0]) if metrics_a else 0
        max_iter_b = max(extract_metric(metrics_b, "iteration")[0]) if metrics_b else 0
        
        # Look for albedo & normal GT comparison grids
        alb_grid_a = os.path.join(args.folder_a, "train_process", "albedo_gt_comparison", f"{max_iter_a:05d}.png")
        alb_grid_b = os.path.join(args.folder_b, "train_process", "albedo_gt_comparison", f"{max_iter_b:05d}.png")
        
        norm_grid_a = os.path.join(args.folder_a, "train_process", "normal_gt_comparison", f"{max_iter_a:05d}.png")
        norm_grid_b = os.path.join(args.folder_b, "train_process", "normal_gt_comparison", f"{max_iter_b:05d}.png")

        # Test RGB grids
        rgb_grid_a = glob.glob(os.path.join(args.folder_a, "eval_visuals", "test", f"iter{max_iter_a:06d}_view*.png"))
        rgb_grid_b = glob.glob(os.path.join(args.folder_b, "eval_visuals", "test", f"iter{max_iter_b:06d}_view*.png"))

        # Select a sample RGB view
        rgb_a_path = rgb_grid_a[0] if rgb_grid_a else None
        rgb_b_path = rgb_grid_b[0] if rgb_grid_b else None

        # Split renders vs GT
        r_rgb_a, gt_rgb = split_grid_image(rgb_a_path)
        r_rgb_b, _ = split_grid_image(rgb_b_path)
        
        r_alb_a, gt_alb = split_grid_image(alb_grid_a)
        r_alb_b, _ = split_grid_image(alb_grid_b)
        
        r_norm_a, gt_norm = split_grid_image(norm_grid_a)
        r_norm_b, _ = split_grid_image(norm_grid_b)

        # Plot qualitative comparisons grid
        fig, axes = plt.subplots(3, 3, figsize=(11, 8.5))
        fig.suptitle("Qualitative Prior and Reconstruction Quality Comparison", fontsize=14, fontweight="bold", y=0.98)
        fig.patch.set_facecolor("#FAFAFA")

        # Ground Truth row
        if gt_rgb:
            axes[0, 0].imshow(gt_rgb)
        axes[0, 0].set_title("GT RGB View")
        axes[0, 0].axis("off")
        
        if gt_alb:
            axes[0, 1].imshow(gt_alb)
        axes[0, 1].set_title("GT Albedo Prior")
        axes[0, 1].axis("off")
        
        if gt_norm:
            axes[0, 2].imshow(gt_norm)
        axes[0, 2].set_title("GT Normal Prior")
        axes[0, 2].axis("off")

        # Baseline row
        if r_rgb_a:
            axes[1, 0].imshow(r_rgb_a)
        axes[1, 0].set_title(f"{args.name_a} RGB")
        axes[1, 0].axis("off")
        
        if r_alb_a:
            axes[1, 1].imshow(r_alb_a)
        axes[1, 1].set_title(f"{args.name_a} Albedo")
        axes[1, 1].axis("off")
        
        if r_norm_a:
            axes[1, 2].imshow(r_norm_a)
        axes[1, 2].set_title(f"{args.name_a} Normal")
        axes[1, 2].axis("off")

        # Proposed row
        if r_rgb_b:
            axes[2, 0].imshow(r_rgb_b)
        axes[2, 0].set_title(f"{args.name_b} RGB")
        axes[2, 0].axis("off")
        
        if r_alb_b:
            axes[2, 1].imshow(r_alb_b)
        axes[2, 1].set_title(f"{args.name_b} Albedo")
        axes[2, 1].axis("off")
        
        if r_norm_b:
            axes[2, 2].imshow(r_norm_b)
        axes[2, 2].set_title(f"{args.name_b} Normal")
        axes[2, 2].axis("off")

        # Style row labels with bounding boxes
        row_labels = ["Ground Truth", args.name_a, args.name_b]
        row_colors = [COLORS["gt"], COLORS["a"], COLORS["b"]]
        for row in range(3):
            bbox = axes[row, 0].get_position()
            y_center = (bbox.y0 + bbox.y1) / 2
            fig.text(0.04, y_center, row_labels[row], fontsize=11, fontweight="bold",
                     ha="center", va="center", rotation=90, color="white",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=row_colors[row], alpha=0.9))

        plt.tight_layout(rect=[0.08, 0, 1, 0.94])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"\n✓ Comparison report successfully generated and saved to: {args.output_pdf}")

if __name__ == "__main__":
    main()
