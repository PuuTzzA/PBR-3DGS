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
    parser.add_argument("--run", action="append", nargs=2, metavar=("FOLDER", "NAME"), help="Output folder and display name for a run (can be specified multiple times)")
    parser.add_argument("--folder_a", type=str, default=None, help="Path to Run A folder (Baseline)")
    parser.add_argument("--folder_b", type=str, default=None, help="Path to Run B folder (Proposed)")
    parser.add_argument("--name_a", type=str, default="Baseline (No Priors)", help="Display name for Run A")
    parser.add_argument("--name_b", type=str, default="Proposed (With Priors)", help="Display name for Run B")
    parser.add_argument("--output_pdf", type=str, default=None, help="Output comparison PDF path")
    args = parser.parse_args()

    runs = []
    if args.run:
        for folder, name in args.run:
            runs.append({"folder": folder, "name": name})
    else:
        # Fallback to --folder_a and --folder_b
        if not args.folder_a and not args.folder_b:
            parser.error("You must specify either --run or --folder_a and --folder_b")
        if args.folder_a:
            runs.append({"folder": args.folder_a, "name": args.name_a})
        if args.folder_b:
            runs.append({"folder": args.folder_b, "name": args.name_b})

    # Configure run colors
    RUN_COLORS = [
        "#2196F3",  # Blue
        "#E91E63",  # Pink/Red
        "#9C27B0",  # Purple
        "#FF9800",  # Orange
        "#00BCD4",  # Cyan
        "#795548",  # Brown
        "#607D8B",  # Blue Grey
    ]
    for idx, run in enumerate(runs):
        run["color"] = RUN_COLORS[idx % len(RUN_COLORS)]

    print(f"Comparing runs:")
    for run in runs:
        print(f"  Run: {run['folder']} ({run['name']})")

    # Default output PDF path if not specified
    if args.output_pdf is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("outputs", f"comparison_{timestamp}")
        args.output_pdf = os.path.join(out_dir, "comparison_report.pdf")

    # Ensure parent directory of output_pdf exists
    out_dir = os.path.dirname(args.output_pdf)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = "."

    # Helper to save a single metric plot as a PNG
    def save_single_metric_plot(key, ylabel, title, filename, higher_better=True):
        fig_single, ax_single = plt.subplots(figsize=(8, 6))
        fig_single.patch.set_facecolor("#FAFAFA")
        
        for run in runs:
            iters, vals = extract_metric(run["metrics"], key)
            if iters:
                label = f"{run['name']} (final: {vals[-1]:.3f})"
                ax_single.plot(iters, vals, "o-", color=run["color"], markersize=3, linewidth=1.5, label=label)
        
        ax_single.set_xlabel("Iteration")
        ax_single.set_ylabel(ylabel)
        ax_single.set_title(title, fontsize=12, fontweight="bold")
        ax_single.grid(True, alpha=0.3, linestyle="--")
        ax_single.legend(loc="best")
        
        fig_single.tight_layout()
        fig_single.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close(fig_single)



    # Load configurations, metrics, and loss logs
    for run in runs:
        folder = run["folder"]
        run["cfg"] = parse_cfg_args(os.path.join(folder, "cfg_args"))
        run["metrics"] = load_metrics_log(folder)

    valid_runs = [r for r in runs if r["metrics"]]
    if not valid_runs:
        print("ERROR: None of the specified folders have a valid metrics_log.json! Cannot generate comparison.")
        sys.exit(1)

    # Resolve dataset paths
    resolved_dataset = "Unknown"
    for run in runs:
        source_path = run["cfg"].get("source_path", "Unknown")
        if source_path != "Unknown":
            resolved_dataset = source_path
            break
    dataset_name = os.path.basename(os.path.normpath(resolved_dataset)) if resolved_dataset else "Unknown"

    # Find common HDRIs evaluated for relighting across all runs
    def get_evaluated_hdris(metrics_log):
        hdris = set()
        for entry in metrics_log:
            for key in entry.keys():
                if key.startswith("relight_") and key.endswith("_psnr"):
                    hdri_name = key[len("relight_"):-len("_psnr")]
                    hdris.add(hdri_name)
        return hdris

    common_hdris = None
    for run in runs:
        run_hdris = get_evaluated_hdris(run["metrics"])
        if common_hdris is None:
            common_hdris = run_hdris
        else:
            common_hdris = common_hdris.intersection(run_hdris)
    common_hdris = sorted(list(common_hdris)) if common_hdris else []
    
    if common_hdris:
        print(f"Found common HDRIs for relighting comparison: {common_hdris}")
    else:
        print("No common HDRIs found for relighting comparison.")

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
                         "lambda_albedo_gt", "lambda_normal_gt", "lambda_metallic_gt", "lambda_roughness_gt",
                         "use_prior_weight_scheduler", "prior_weight_scheduler_ratio"]
        all_keys = set()
        for run in runs:
            all_keys.update(run["cfg"].keys())
        all_keys = sorted(list(all_keys))
        
        # Check if values differ across any of the runs
        diff_keys = []
        for k in all_keys:
            vals = [run["cfg"].get(k) for run in runs]
            if len(set(vals)) > 1:
                diff_keys.append(k)
        
        # Merge critical keys and differences
        keys_to_show = sorted(list(set(critical_keys + diff_keys)))
        
        table_data = []
        for k in keys_to_show:
            row = []
            is_diff = False
            first_val = runs[0]["cfg"].get(k, "—")
            for run in runs:
                val = run["cfg"].get(k, "—")
                if val != first_val:
                    is_diff = True
                row.append(str(val))
            marker = "*" if is_diff else ""
            table_data.append([f"{k}{marker}"] + row)

        header = ["Argument"] + [run["name"] for run in runs]
        num_runs = len(runs)
        col_widths = [0.3] + [0.6 / num_runs] * num_runs
        table = ax.table(cellText=table_data, colLabels=header, loc="center", cellLoc="left",
                         colWidths=col_widths)
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
        for hdri in common_hdris:
            metrics_def.append((f"relight_{hdri}_psnr", f"Relight PSNR ({hdri}) (dB) ↑", True))
            metrics_def.append((f"relight_{hdri}_ssim", f"Relight SSIM ({hdri}) ↑", True))

        metrics_table_data = []
        for key, name, higher_better in metrics_def:
            row = []
            vals = []
            for run in runs:
                val = get_final_metrics(run["metrics"]).get(key, None)
                vals.append(val)
                row.append(f"{val:.4f}" if val is not None else "N/A")
            
            # Determine winner
            winner = ""
            valid_vals = [v for v in vals if v is not None]
            if len(valid_vals) > 0:
                if len(set(valid_vals)) > 1:
                    best_val = max(valid_vals) if higher_better else min(valid_vals)
                    best_idx = vals.index(best_val)
                    winner = runs[best_idx]["name"]
                else:
                    winner = "Tie" if len(valid_vals) > 1 else ""
            
            metrics_table_data.append([name] + row + [winner])

        m_header = ["Evaluation Metric"] + [run["name"] for run in runs] + ["Top Performer"]
        col_widths = [0.35] + [0.44 / num_runs] * num_runs + [0.21]
        m_table = ax.table(cellText=metrics_table_data, colLabels=m_header, loc="center", cellLoc="center",
                           colWidths=col_widths)
        m_table.auto_set_font_size(False)
        m_table.set_fontsize(10)
        
        # Dynamically scale vertical scaling to prevent table overlap with more rows
        table_scale_y = max(1.0, 1.8 - 0.05 * (len(metrics_table_data) - 5))
        m_table.scale(1.0, table_scale_y)

        # Style metrics table
        for j in range(len(m_header)):
            m_table[0, j].set_facecolor("#16213E")
            m_table[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(metrics_table_data) + 1):
            winner_name = metrics_table_data[i-1][-1]
            for j in range(len(m_header)):
                if winner_name != "Tie" and winner_name != "" and m_header[j] == winner_name:
                    m_table[i, j].set_facecolor("#C8E6C9")  # Light green for winner
                    m_table[i, j].set_text_props(fontweight="bold")
                else:
                    m_table[i, j].set_facecolor("#F5F5F5" if i % 2 == 0 else "white")

        # Quantitative text findings - shift down dynamically based on table size
        findings_y = max(0.05, 0.18 - 0.015 * len(common_hdris))
        ax.text(0.05, findings_y, "Key Findings & Analysis:", fontsize=12, fontweight="bold",
                ha="left", va="top", transform=ax.transAxes, color="#1A1A2E")
        
        analysis_text = ""
        if len(runs) >= 2:
            run_a, run_b = runs[0], runs[1]
            final_a = get_final_metrics(run_a["metrics"])
            final_b = get_final_metrics(run_b["metrics"])
            
            alb_a = final_a.get("test_albedo_psnr")
            alb_b = final_b.get("test_albedo_psnr")
            if alb_a is not None and alb_b is not None:
                diff = alb_b - alb_a
                analysis_text += f"• **Albedo Reconstruction Accuracy:** {run_b['name']} achieves an albedo PSNR of **{alb_b:.2f} dB**, which is an improvement of **{diff:+.2f} dB** over {run_a['name']} (**{alb_a:.2f} dB**).\n"
            
            ang_a = final_a.get("test_normal_ang_err")
            ang_b = final_b.get("test_normal_ang_err")
            if ang_a is not None and ang_b is not None:
                diff = ang_a - ang_b
                analysis_text += f"• **Surface Normal Estimation:** The angular error of the estimated surface normals drops from **{ang_a:.2f}°** to **{ang_b:.2f}°** (**{diff:+.2f}°** reduction) for {run_b['name']} vs {run_a['name']}.\n"
                
            psnr_a = final_a.get("test_psnr")
            psnr_b = final_b.get("test_psnr")
            if psnr_a is not None and psnr_b is not None:
                diff = psnr_b - psnr_a
                analysis_text += f"• **Overall RGB Novel-View Synthesis:** Novel-view synthesis PSNR changes by **{diff:+.2f} dB** ({run_a['name']}: **{psnr_a:.2f} dB** vs {run_b['name']}: **{psnr_b:.2f} dB**).\n"

            for hdri in common_hdris:
                r_psnr_a = final_a.get(f"relight_{hdri}_psnr")
                r_psnr_b = final_b.get(f"relight_{hdri}_psnr")
                if r_psnr_a is not None and r_psnr_b is not None:
                    diff = r_psnr_b - r_psnr_a
                    analysis_text += f"• **Relighting under {hdri}:** PSNR changes by **{diff:+.2f} dB** ({run_a['name']}: **{r_psnr_a:.2f} dB** vs {run_b['name']}: **{r_psnr_b:.2f} dB**).\n"
        else:
            analysis_text = "• Comparative metrics show training progress for the loaded run."
            
        if not analysis_text:
            analysis_text = "• Comparative metrics show training progress across runs."

        ax.text(0.05, findings_y - 0.05, analysis_text, fontsize=10, ha="left", va="top",
                transform=ax.transAxes, color="#333", linespacing=1.6)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ─── Page 3: Overlaid Metrics Comparison Plots ───
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Convergence and Performance Curves Over Training", fontsize=14, fontweight="bold", y=0.98)
        fig.patch.set_facecolor("#FAFAFA")

        def plot_comparison_curve_n(ax, key, ylabel, title, higher_better=True):
            for run in runs:
                iters, vals = extract_metric(run["metrics"], key)
                if iters:
                    label = f"{run['name']} (final: {vals[-1]:.3f})"
                    ax.plot(iters, vals, "o-", color=run["color"], markersize=2, linewidth=1.2, label=label)
            
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc="best")

        plot_comparison_curve_n(axes[0, 0], "test_psnr", "PSNR (dB)", "Novel-View RGB PSNR ↑", higher_better=True)
        plot_comparison_curve_n(axes[0, 1], "test_albedo_psnr", "PSNR (dB)", "Albedo PSNR ↑", higher_better=True)
        plot_comparison_curve_n(axes[1, 0], "test_normal_ang_err", "Degrees (°)", "Normal Angular Error ↓", higher_better=False)
        plot_comparison_curve_n(axes[1, 1], "train_loss", "Loss (EMA)", "Training Loss ↓", higher_better=False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Save individual PNGs for convergence curves
        save_single_metric_plot("test_psnr", "PSNR (dB)", "Novel-View RGB PSNR", "novel_view_rgb_psnr.png", higher_better=True)
        save_single_metric_plot("test_albedo_psnr", "PSNR (dB)", "Albedo Prior PSNR", "albedo_psnr.png", higher_better=True)
        save_single_metric_plot("test_normal_ang_err", "Degrees (°)", "Normal Angular Error", "normal_angular_error.png", higher_better=False)
        save_single_metric_plot("train_loss", "Loss (EMA)", "Training Loss", "training_loss.png", higher_better=False)

        # ─── Page 3b: Relighting Curves (if common HDRIs exist) ───
        if common_hdris:
            fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
            fig.suptitle("Relighting Performance Curves Over Training", fontsize=14, fontweight="bold", y=0.98)
            fig.patch.set_facecolor("#FAFAFA")
            
            marker_styles = ["o", "s", "^", "v", "<", ">", "d"]
            # PSNR Plot
            ax_psnr = axes[0]
            for run in runs:
                for h_idx, hdri in enumerate(common_hdris):
                    iters, vals = extract_metric(run["metrics"], f"relight_{hdri}_psnr")
                    if iters:
                        label = f"{run['name']} - {hdri} (final: {vals[-1]:.2f})"
                        marker = marker_styles[h_idx % len(marker_styles)]
                        ax_psnr.plot(iters, vals, marker=marker, linestyle="-", color=run["color"], markersize=4, linewidth=1.2, label=label)
            ax_psnr.set_xlabel("Iteration")
            ax_psnr.set_ylabel("PSNR (dB)")
            ax_psnr.set_title("Relighting PSNR ↑")
            ax_psnr.legend(loc="best", fontsize=8)
            ax_psnr.grid(True, alpha=0.3, linestyle="--")

            # SSIM Plot
            ax_ssim = axes[1]
            for run in runs:
                for h_idx, hdri in enumerate(common_hdris):
                    iters, vals = extract_metric(run["metrics"], f"relight_{hdri}_ssim")
                    if iters:
                        label = f"{run['name']} - {hdri} (final: {vals[-1]:.4f})"
                        marker = marker_styles[h_idx % len(marker_styles)]
                        ax_ssim.plot(iters, vals, marker=marker, linestyle="-", color=run["color"], markersize=4, linewidth=1.2, label=label)
            ax_ssim.set_xlabel("Iteration")
            ax_ssim.set_ylabel("SSIM")
            ax_ssim.set_title("Relighting SSIM ↑")
            ax_ssim.legend(loc="best", fontsize=8)
            ax_ssim.grid(True, alpha=0.3, linestyle="--")

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Save individual PNGs for common HDRIs
            for hdri in common_hdris:
                save_single_metric_plot(f"relight_{hdri}_psnr", "PSNR (dB)", f"Relighting PSNR under {hdri}", f"relight_{hdri}_psnr.png", higher_better=True)
                save_single_metric_plot(f"relight_{hdri}_ssim", "SSIM", f"Relighting SSIM under {hdri}", f"relight_{hdri}_ssim.png", higher_better=True)

        # ─── Page 4: Visual Reconstruction Comparison Grid (RGB, Albedo, Normal) ───
        print("Locating visual comparisons...")
        
        max_iters = []
        for run in runs:
            iters, _ = extract_metric(run["metrics"], "iteration")
            max_iters.append(max(iters) if iters else 0)

        # Collect split images for all runs
        run_visuals = []
        gt_rgb, gt_alb, gt_norm = None, None, None

        for run, max_iter in zip(runs, max_iters):
            folder = run["folder"]
            alb_grid = os.path.join(folder, "train_process", "albedo_gt_comparison", f"{max_iter:05d}.png")
            norm_grid = os.path.join(folder, "train_process", "normal_gt_comparison", f"{max_iter:05d}.png")
            
            rgb_grids = glob.glob(os.path.join(folder, "eval_visuals", "test", f"iter{max_iter:06d}_view*.png"))
            rgb_path = rgb_grids[0] if rgb_grids else None

            r_rgb, current_gt_rgb = split_grid_image(rgb_path)
            r_alb, current_gt_alb = split_grid_image(alb_grid)
            r_norm, current_gt_norm = split_grid_image(norm_grid)

            if gt_rgb is None and current_gt_rgb is not None:
                gt_rgb = current_gt_rgb
            if gt_alb is None and current_gt_alb is not None:
                gt_alb = current_gt_alb
            if gt_norm is None and current_gt_norm is not None:
                gt_norm = current_gt_norm

            run_visuals.append({
                "name": run["name"],
                "color": run["color"],
                "rgb": r_rgb,
                "alb": r_alb,
                "norm": r_norm
            })

        # Plot qualitative comparisons grid
        num_rows = 1 + len(runs)
        fig_height = 2.5 * num_rows
        fig, axes = plt.subplots(num_rows, 3, figsize=(11, fig_height))
        
        # Ensure axes is always a 2D array
        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)

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

        # Run rows
        for i, vis in enumerate(run_visuals):
            row_idx = i + 1
            
            if vis["rgb"]:
                axes[row_idx, 0].imshow(vis["rgb"])
            axes[row_idx, 0].set_title(f"{vis['name']} RGB")
            axes[row_idx, 0].axis("off")
            
            if vis["alb"]:
                axes[row_idx, 1].imshow(vis["alb"])
            axes[row_idx, 1].set_title(f"{vis['name']} Albedo")
            axes[row_idx, 1].axis("off")
            
            if vis["norm"]:
                axes[row_idx, 2].imshow(vis["norm"])
            axes[row_idx, 2].set_title(f"{vis['name']} Normal")
            axes[row_idx, 2].axis("off")

        # Style row labels with bounding boxes
        row_labels = ["Ground Truth"] + [vis["name"] for vis in run_visuals]
        row_colors = ["#4CAF50"] + [vis["color"] for vis in run_visuals]
        for row in range(num_rows):
            bbox = axes[row, 0].get_position()
            y_center = (bbox.y0 + bbox.y1) / 2
            fig.text(0.04, y_center, row_labels[row], fontsize=11, fontweight="bold",
                     ha="center", va="center", rotation=90, color="white",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=row_colors[row], alpha=0.9))

        plt.tight_layout(rect=[0.08, 0, 1, 0.94])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ─── Page 5+: Relighting Visual Comparisons (per common HDRI) ───
        for hdri in common_hdris:
            # Let's find the views evaluated under this HDRI for the last iteration
            # We can inspect the first run's folder to find the available view names
            first_run = runs[0]
            first_max_iter = max_iters[0]
            hdri_vis_path = os.path.join(first_run["folder"], "eval_visuals", f"relight_{hdri}")
            
            # Find render files
            render_pattern = os.path.join(hdri_vis_path, "*_render.png")
            render_files = sorted(glob.glob(render_pattern))
            
            # Extract view names for the latest iteration in the folder
            view_names = []
            if render_files:
                # Find the largest iteration number present in the filenames
                iters_found = []
                for rf in render_files:
                    match_iter = re.search(r"iter(\d+)_", os.path.basename(rf))
                    if match_iter:
                        iters_found.append(int(match_iter.group(1)))
                latest_iter = max(iters_found) if iters_found else first_max_iter
                
                # Filter render files to only those from the latest iteration
                latest_pattern = f"iter{latest_iter:06d}_"
                for rf in render_files:
                    if latest_pattern in os.path.basename(rf):
                        basename = os.path.basename(rf)
                        match = re.match(r"iter\d+_(.*)_render\.png", basename)
                        if match:
                            view_names.append(match.group(1))
            
            # Limit to at most 3 views to avoid extremely long reports
            view_names = view_names[:3]
            
            if not view_names:
                continue
                
            num_views = len(view_names)
            num_cols = 1 + len(runs) # GT + number of runs
            fig_height = 2.5 * num_views
            fig, axes = plt.subplots(num_views, num_cols, figsize=(11, fig_height))
            
            # Ensure axes is 2D
            if num_views == 1:
                axes = np.expand_dims(axes, axis=0)
            if num_cols == 1:
                axes = np.expand_dims(axes, axis=-1)
                
            fig.suptitle(f"Relighting Quality Comparison: {hdri}", fontsize=14, fontweight="bold", y=0.98)
            fig.patch.set_facecolor("#FAFAFA")
            
            for v_idx, view_name in enumerate(view_names):
                # 1. Load Ground Truth
                # We search across all runs to find the GT image for this view under this HDRI
                gt_image = None
                for run, max_iter in zip(runs, max_iters):
                    gt_path = os.path.join(run["folder"], "eval_visuals", f"relight_{hdri}", f"iter{max_iter:06d}_{view_name}_gt.png")
                    if os.path.exists(gt_path):
                        try:
                            gt_image = Image.open(gt_path)
                            break
                        except Exception:
                            pass
                
                # Plot GT in the first column
                ax_gt = axes[v_idx, 0]
                if gt_image is not None:
                    ax_gt.imshow(gt_image)
                else:
                    ax_gt.text(0.5, 0.5, "GT N/A", ha="center", va="center", transform=ax_gt.transAxes, color="gray", fontsize=12)
                ax_gt.set_title(f"GT — {view_name}")
                ax_gt.axis("off")
                
                # 2. Load and plot each run's rendering
                for r_idx, (run, max_iter) in enumerate(zip(runs, max_iters)):
                    rend_path = os.path.join(run["folder"], "eval_visuals", f"relight_{hdri}", f"iter{max_iter:06d}_{view_name}_render.png")
                    rend_image = None
                    if os.path.exists(rend_path):
                        try:
                            rend_image = Image.open(rend_path)
                        except Exception:
                            pass
                            
                    ax_rend = axes[v_idx, 1 + r_idx]
                    if rend_image is not None:
                        ax_rend.imshow(rend_image)
                    else:
                        ax_rend.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax_rend.transAxes, color="gray", fontsize=12)
                    ax_rend.set_title(f"{run['name']} — {view_name}")
                    ax_rend.axis("off")
            
            # Add row labels
            for v_idx, view_name in enumerate(view_names):
                bbox = axes[v_idx, 0].get_position()
                y_center = (bbox.y0 + bbox.y1) / 2
                fig.text(0.04, y_center, view_name, fontsize=10, fontweight="bold",
                         ha="center", va="center", rotation=90, color="white",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="#37474F", alpha=0.9))
                         
            plt.tight_layout(rect=[0.08, 0, 1, 0.94])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"\n✓ Comparison report successfully generated and saved to: {args.output_pdf}")

if __name__ == "__main__":
    main()
