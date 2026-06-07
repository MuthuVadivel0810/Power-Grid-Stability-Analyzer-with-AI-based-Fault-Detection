"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: Phase 4 Plotter  (ml/phase4_plotter.py)

  Plots produced:
    1. Model Comparison Bar Chart     — all 5 models, test vs CV acc
    2. Confusion Matrix (normalised)  — best model
    3. ROC Curves (all 5 classes)     — AUC per fault type
    4. Feature Importance             — top-20 Random Forest features
    5. Learning Curve                 — train vs val vs training size
    6. Precision-Recall Curves        — per fault class
    7. Combined Phase 4 Dashboard     — 6-panel grid
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

DARK_BG  = "#1A1A2E"
PANEL_BG = "#16213E"
WHITE    = "#ECEFF4"
RED      = "#E74C3C"
GREEN    = "#2ECC71"
BLUE     = "#3498DB"
YELLOW   = "#F9CA24"
ORANGE   = "#E67E22"
PURPLE   = "#9B59B6"
CYAN     = "#1ABC9C"
GRAY     = "#7F8C8D"

CLASS_COLORS = [GREEN, RED, BLUE, YELLOW, ORANGE]
CLASS_NAMES  = {0:"No Fault", 1:"3-Phase", 2:"Line-Ground",
                3:"Line-Line", 4:"DLine-Ground"}


class Phase4Plotter:

    def __init__(self):
        plt.rcParams.update({
            "font.family"    : "DejaVu Sans",
            "axes.titlecolor": WHITE,
            "axes.labelcolor": WHITE,
            "xtick.color"    : WHITE,
            "ytick.color"    : WHITE,
        })

    def _style(self, ax, title="", xlabel="", ylabel="", legend=True):
        ax.set_facecolor(PANEL_BG)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["bottom", "left"]:
            ax.spines[s].set_color("#555")
        ax.grid(True, linestyle="--", lw=0.5, alpha=0.35, color="#aaa")
        if title:  ax.set_title(title,  color=WHITE, fontsize=10, fontweight="bold", pad=8)
        if xlabel: ax.set_xlabel(xlabel, color=WHITE, fontsize=8)
        if ylabel: ax.set_ylabel(ylabel, color=WHITE, fontsize=8)

    # ==================================================================
    # 1.  MODEL COMPARISON
    # ==================================================================
    def plot_model_comparison(self, results: dict, save_path=None):
        names    = list(results.keys())
        test_acc = [results[n]["test_acc"] for n in names]
        cv_mean  = [results[n]["cv_mean"]  for n in names]
        cv_std   = [results[n]["cv_std"]   for n in names]

        x = np.arange(len(names))
        w = 0.35

        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor(DARK_BG)

        b1 = ax.bar(x - w/2, test_acc, w, label="Test Accuracy",
                    color=BLUE, edgecolor="white", lw=0.7, alpha=0.9)
        b2 = ax.bar(x + w/2, cv_mean, w, label="CV Mean (5-fold)",
                    color=GREEN, edgecolor="white", lw=0.7, alpha=0.9,
                    yerr=cv_std, capsize=5,
                    error_kw={"ecolor": WHITE, "lw": 1.5})

        for bar, val in zip(b1, test_acc):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.15,
                    f"{val:.2f}%", ha="center", va="bottom",
                    color=WHITE, fontsize=8, fontweight="bold")
        for bar, val in zip(b2, cv_mean):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.15,
                    f"{val:.2f}%", ha="center", va="bottom",
                    color=WHITE, fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_", "\n") for n in names],
                           color=WHITE, fontsize=8)
        ax.set_ylim(80, 102)
        ax.axhline(y=95, color=YELLOW, lw=1.2, ls="--",
                   label="95% threshold")
        ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=9)
        self._style(ax,
            title="Model Comparison — Test Accuracy vs 5-Fold CV Accuracy",
            xlabel="Model", ylabel="Accuracy (%)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=DARK_BG)
            print(f"[Phase4Plotter] Model comparison saved → {save_path}")
        plt.close()

    # ==================================================================
    # 2.  CONFUSION MATRIX
    # ==================================================================
    def plot_confusion_matrix(self, cm_norm, cm_raw, save_path=None):
        labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
        n      = len(labels)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor(DARK_BG)

        for ax, data, title, fmt in zip(
            axes,
            [cm_raw,  cm_norm],
            ["Confusion Matrix — Raw Counts",
             "Confusion Matrix — Normalised (Recall)"],
            [".0f",   ".3f"]
        ):
            ax.set_facecolor(PANEL_BG)
            im = ax.imshow(data, cmap="Blues", vmin=0,
                           vmax=(1.0 if "Norm" in title else None))
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.yaxis.set_tick_params(color=WHITE)
            cbar.ax.set_yticklabels(
                [f"{v:.1f}" for v in cbar.get_ticks()], color=WHITE)

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=30, ha="right",
                               color=WHITE, fontsize=9)
            ax.set_yticklabels(labels, color=WHITE, fontsize=9)
            ax.set_xlabel("Predicted", color=WHITE, fontsize=9)
            ax.set_ylabel("True",      color=WHITE, fontsize=9)
            ax.set_title(title, color=WHITE, fontsize=11,
                         fontweight="bold", pad=10)

            thresh = data.max() / 2.0
            for i in range(n):
                for j in range(n):
                    val = data[i, j]
                    txt = format(val, fmt)
                    color = "white" if val < thresh else "#111"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=10, fontweight="bold", color=color)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=DARK_BG)
            print(f"[Phase4Plotter] Confusion matrix saved → {save_path}")
        plt.close()

    # ==================================================================
    # 3.  ROC CURVES
    # ==================================================================
    def plot_roc_curves(self, roc_data: dict, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(DARK_BG)

        for i, (cls, d) in enumerate(roc_data.items()):
            ax.plot(d["fpr"], d["tpr"],
                    color=CLASS_COLORS[i], lw=2.0,
                    label=f"{CLASS_NAMES[cls]}  (AUC={d['auc']:.4f})",
                    alpha=0.9)

        ax.plot([0, 1], [0, 1], color=GRAY, lw=1.2,
                linestyle="--", label="Random (AUC=0.5)")
        ax.fill_between([0, 1], [0, 0], [1, 1], alpha=0.04, color=GRAY)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=9,
                  loc="lower right")
        self._style(ax,
            title="ROC Curves — One-vs-Rest (5 Fault Classes)",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate (Recall)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=DARK_BG)
            print(f"[Phase4Plotter] ROC curves saved → {save_path}")
        plt.close()

    # ==================================================================
    # 4.  FEATURE IMPORTANCE
    # ==================================================================
    def plot_feature_importance(self, importance_dict: dict,
                                top_n: int = 20, save_path=None):
        items   = list(importance_dict.items())[:top_n]
        feats   = [k for k, _ in items]
        vals    = [v["importance"] for _, v in items]
        stds    = [v["std"]        for _, v in items]

        colors = [RED if v > 0.08 else BLUE if v > 0.04 else GREEN
                  for v in vals]

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(DARK_BG)

        y = np.arange(len(feats))
        bars = ax.barh(y, vals, xerr=stds, color=colors,
                       edgecolor="white", lw=0.5, alpha=0.88,
                       error_kw={"ecolor": WHITE, "lw": 1.2,
                                 "capsize": 4})

        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", ha="left",
                    color=WHITE, fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels(
            [f.replace("_", " ") for f in feats],
            color=WHITE, fontsize=8)
        ax.invert_yaxis()

        legend_patches = [
            mpatches.Patch(color=RED,   label="High  (>8%)"),
            mpatches.Patch(color=BLUE,  label="Medium (4–8%)"),
            mpatches.Patch(color=GREEN, label="Low    (<4%)"),
        ]
        ax.legend(handles=legend_patches, facecolor=PANEL_BG,
                  labelcolor=WHITE, fontsize=8)
        self._style(ax,
            title=f"Top-{top_n} Feature Importances — Random Forest",
            xlabel="Importance Score (mean decrease in impurity)",
            ylabel="Feature")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=DARK_BG)
            print(f"[Phase4Plotter] Feature importance saved → {save_path}")
        plt.close()

    # ==================================================================
    # 5.  LEARNING CURVE
    # ==================================================================
    def plot_learning_curve(self, lc_data: dict, save_path=None):
        ts  = lc_data["train_sizes"]
        tm  = lc_data["train_mean"]
        ts_ = lc_data["train_std"]
        vm  = lc_data["val_mean"]
        vs  = lc_data["val_std"]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(DARK_BG)

        ax.plot(ts, tm * 100, "o-", color=BLUE,  lw=2.0,
                label="Training Accuracy",   markersize=6)
        ax.fill_between(ts,
                        (tm - ts_) * 100,
                        (tm + ts_) * 100,
                        alpha=0.2, color=BLUE)

        ax.plot(ts, vm * 100, "s-", color=GREEN, lw=2.0,
                label="Validation Accuracy", markersize=6)
        ax.fill_between(ts,
                        (vm - vs) * 100,
                        (vm + vs) * 100,
                        alpha=0.2, color=GREEN)

        ax.axhline(y=95, color=YELLOW, lw=1.2, ls="--",
                   alpha=0.8, label="95% baseline")
        ax.set_ylim(70, 103)
        ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=9)
        self._style(ax,
            title="Learning Curve — Training Size vs Accuracy",
            xlabel="Training Set Size (samples)",
            ylabel="Accuracy (%)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=DARK_BG)
            print(f"[Phase4Plotter] Learning curve saved → {save_path}")
        plt.close()

    # ==================================================================
    # 6.  PRECISION-RECALL CURVES
    # ==================================================================
    def plot_pr_curves(self, pr_data: dict, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(DARK_BG)

        for i, (cls, d) in enumerate(pr_data.items()):
            ax.plot(d["recall"], d["precision"],
                    color=CLASS_COLORS[i], lw=2.0,
                    label=f"{CLASS_NAMES[cls]}  (AP={d['ap']:.4f})",
                    alpha=0.9)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=9)
        self._style(ax,
            title="Precision-Recall Curves — All Fault Classes",
            xlabel="Recall", ylabel="Precision")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=DARK_BG)
            print(f"[Phase4Plotter] PR curves saved → {save_path}")
        plt.close()

    # ==================================================================
    # 7.  COMBINED DASHBOARD
    # ==================================================================
    def plot_dashboard(self, results, metrics, importance_dict,
                       best_name, save_path=None):
        fig = plt.figure(figsize=(22, 16))
        fig.patch.set_facecolor(DARK_BG)
        gs  = gridspec.GridSpec(3, 3, figure=fig,
                                hspace=0.55, wspace=0.42)

        ax1 = fig.add_subplot(gs[0, :2])   # Model comparison (wide)
        ax2 = fig.add_subplot(gs[0, 2])    # Summary text
        ax3 = fig.add_subplot(gs[1, :2])   # Confusion matrix normalised
        ax4 = fig.add_subplot(gs[1, 2])    # ROC single-axis (zoomed)
        ax5 = fig.add_subplot(gs[2, :2])   # Feature importance
        ax6 = fig.add_subplot(gs[2, 2])    # Learning curve

        # ── Model comparison ──────────────────────────────────────────
        names    = list(results.keys())
        test_acc = [results[n]["test_acc"] for n in names]
        cv_mean  = [results[n]["cv_mean"]  for n in names]
        x = np.arange(len(names)); w = 0.35
        ax1.set_facecolor(PANEL_BG)
        b1 = ax1.bar(x-w/2, test_acc, w, color=BLUE,  alpha=0.9,
                     edgecolor="white", lw=0.6, label="Test Acc")
        b2 = ax1.bar(x+w/2, cv_mean,  w, color=GREEN, alpha=0.9,
                     edgecolor="white", lw=0.6, label="CV Mean")
        for bar, v in zip(list(b1)+list(b2), test_acc+cv_mean):
            ax1.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.1, f"{v:.1f}",
                     ha="center", va="bottom", color=WHITE, fontsize=7)
        ax1.set_xticks(x)
        ax1.set_xticklabels([n.replace("_","\n") for n in names],
                            color=WHITE, fontsize=7)
        ax1.set_ylim(80, 103)
        ax1.axhline(95, color=YELLOW, lw=1, ls="--")
        ax1.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=8)
        self._style(ax1, title="Model Comparison",
                    xlabel="Model", ylabel="Accuracy (%)")

        # ── Summary text ──────────────────────────────────────────────
        ax2.set_facecolor(PANEL_BG); ax2.axis("off")
        acc   = metrics["accuracy"] * 100
        f1    = metrics["f1_macro"]
        roc_d = metrics["roc"]
        avg_auc = np.mean([d["auc"] for d in roc_d.values()]) if roc_d else 0
        text = (
            f"🏆  BEST MODEL\n"
            f"{'─'*22}\n"
            f"{best_name.replace('_',' ')}\n\n"
            f"Test Accuracy\n"
            f"  {acc:.3f}%\n\n"
            f"Macro F1-Score\n"
            f"  {f1:.4f}\n\n"
            f"Mean ROC-AUC\n"
            f"  {avg_auc:.4f}\n\n"
            f"Classes: 5\n"
            f"Features: 32"
        )
        ax2.text(0.05, 0.95, text, transform=ax2.transAxes,
                 fontsize=10, va="top", color=WHITE,
                 fontfamily="monospace",
                 bbox=dict(facecolor="#0F3460", alpha=0.7,
                           boxstyle="round,pad=0.5"))

        # ── Confusion matrix ──────────────────────────────────────────
        cm_norm = metrics["cm_norm"]
        labels  = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
        ax3.set_facecolor(PANEL_BG)
        im = ax3.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax3.set_xticks(range(5)); ax3.set_yticks(range(5))
        ax3.set_xticklabels(labels, rotation=25, ha="right",
                            color=WHITE, fontsize=7)
        ax3.set_yticklabels(labels, color=WHITE, fontsize=7)
        ax3.set_xlabel("Predicted", color=WHITE, fontsize=8)
        ax3.set_ylabel("True",      color=WHITE, fontsize=8)
        ax3.set_title("Confusion Matrix (Normalised)",
                      color=WHITE, fontsize=10, fontweight="bold")
        for i in range(5):
            for j in range(5):
                v = cm_norm[i, j]
                c = "white" if v < 0.5 else "#111"
                ax3.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=9, fontweight="bold", color=c)

        # ── ROC curves (compact) ──────────────────────────────────────
        ax4.set_facecolor(PANEL_BG)
        for i, (cls, d) in enumerate(roc_d.items()):
            ax4.plot(d["fpr"], d["tpr"], color=CLASS_COLORS[i],
                     lw=1.8, alpha=0.9,
                     label=f"{CLASS_NAMES[cls][:5]} {d['auc']:.3f}")
        ax4.plot([0,1],[0,1], color=GRAY, lw=1, ls="--")
        ax4.set_xlim([0,1]); ax4.set_ylim([0,1.02])
        ax4.legend(facecolor=PANEL_BG, labelcolor=WHITE,
                   fontsize=6.5, loc="lower right")
        self._style(ax4, title="ROC Curves",
                    xlabel="FPR", ylabel="TPR")

        # ── Feature importance ────────────────────────────────────────
        ax5.set_facecolor(PANEL_BG)
        top15   = list(importance_dict.items())[:15]
        f_names = [k.replace("_"," ") for k, _ in top15]
        f_vals  = [v["importance"] for _, v in top15]
        f_std   = [v["std"]        for _, v in top15]
        f_cols  = [RED if v>0.08 else BLUE if v>0.04 else GREEN
                   for v in f_vals]
        y = np.arange(len(f_names))
        ax5.barh(y, f_vals, xerr=f_std, color=f_cols,
                 edgecolor="white", lw=0.5, alpha=0.88,
                 error_kw={"ecolor":WHITE,"lw":1,"capsize":3})
        ax5.set_yticks(y)
        ax5.set_yticklabels(f_names, color=WHITE, fontsize=7)
        ax5.invert_yaxis()
        self._style(ax5, title="Top-15 Feature Importances (RF)",
                    xlabel="Importance", ylabel="Feature")

        # ── Learning curve ────────────────────────────────────────────
        lc  = metrics["learning_curve"]
        ts  = lc["train_sizes"]
        ax6.set_facecolor(PANEL_BG)
        ax6.plot(ts, lc["train_mean"]*100, "o-", color=BLUE,  lw=1.8,
                 label="Train", markersize=4)
        ax6.fill_between(ts,
                         (lc["train_mean"]-lc["train_std"])*100,
                         (lc["train_mean"]+lc["train_std"])*100,
                         alpha=0.2, color=BLUE)
        ax6.plot(ts, lc["val_mean"]*100, "s-", color=GREEN, lw=1.8,
                 label="Val",   markersize=4)
        ax6.fill_between(ts,
                         (lc["val_mean"]-lc["val_std"])*100,
                         (lc["val_mean"]+lc["val_std"])*100,
                         alpha=0.2, color=GREEN)
        ax6.axhline(95, color=YELLOW, lw=1, ls="--")
        ax6.set_ylim(70, 103)
        ax6.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=8)
        self._style(ax6, title="Learning Curve",
                    xlabel="Train size", ylabel="Accuracy (%)")

        fig.suptitle(
            "⚡  ML Fault Classifier Dashboard  —  Phase 4 Results",
            color=WHITE, fontsize=15, fontweight="bold", y=1.01)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=DARK_BG)
            print(f"[Phase4Plotter] Dashboard saved → {save_path}")
        plt.close()