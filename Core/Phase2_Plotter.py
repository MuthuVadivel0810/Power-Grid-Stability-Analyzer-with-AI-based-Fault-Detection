"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: Phase 2 Plotter (core/phase2_plotter.py)

  Plots produced:
    1. NR Convergence curve   — mismatch vs iteration
    2. Bus voltage profile    — |V| at each bus (bar chart)
    3. Bus angle profile      — θ at each bus
    4. Line power flow chart  — P flow on each line
    5. Line losses chart      — P losses per line
    6. Combined dashboard     — all 5 in one figure
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


# ── Colour palette ────────────────────────────────────────────────────────────
DARK_BG   = '#1A1A2E'
PANEL_BG  = '#16213E'
ACCENT1   = '#E94560'
ACCENT2   = '#0F3460'
GREEN     = '#2ECC71'
YELLOW    = '#F9CA24'
BLUE      = '#3498DB'
ORANGE    = '#E67E22'
WHITE     = '#ECEFF4'


class Phase2Plotter:
    """
    All Phase 2 visualization functions.

    Parameters
    ----------
    bus_results  : pd.DataFrame – from ResultsAnalyzer.get_bus_results()
    line_results : pd.DataFrame – from ResultsAnalyzer.get_line_flows()
    nr_result    : dict         – raw NR solver output
    summary      : dict         – from ResultsAnalyzer.get_system_summary()
    """

    def __init__(self, bus_results, line_results, nr_result, summary):
        self.bus_res  = bus_results
        self.line_res = line_results
        self.nr       = nr_result
        self.summary  = summary
        plt.rcParams.update({
            'font.family'     : 'DejaVu Sans',
            'axes.titlecolor' : WHITE,
            'axes.labelcolor' : WHITE,
            'xtick.color'     : WHITE,
            'ytick.color'     : WHITE,
        })

    # ------------------------------------------------------------------
    def _style_ax(self, ax, title: str = "", xlabel: str = "",
                  ylabel: str = ""):
        ax.set_facecolor(PANEL_BG)
        ax.spines['bottom'].set_color('#555')
        ax.spines['left'].set_color('#555')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color='#aaa')
        if title:   ax.set_title(title,   color=WHITE, fontsize=11, fontweight='bold', pad=10)
        if xlabel:  ax.set_xlabel(xlabel, color=WHITE, fontsize=9)
        if ylabel:  ax.set_ylabel(ylabel, color=WHITE, fontsize=9)

    # ==================================================================
    # PLOT 1: NR CONVERGENCE CURVE
    # ==================================================================
    def plot_convergence(self, ax=None, save_path: str = None):
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor(DARK_BG)

        iters = list(range(1, len(self.nr['mismatch_history']) + 1))
        mismatch = self.nr['mismatch_history']

        ax.semilogy(iters, mismatch, 'o-', color=ACCENT1,
                    linewidth=2.5, markersize=8, markerfacecolor=YELLOW,
                    markeredgecolor=ACCENT1, label='Max Mismatch')
        ax.axhline(y=1e-6, color=GREEN, linewidth=1.5,
                   linestyle='--', label='Convergence Threshold (1×10⁻⁶)')

        # Annotate final point
        ax.annotate(
            f'Converged!\n{mismatch[-1]:.2e} pu',
            xy=(iters[-1], mismatch[-1]),
            xytext=(iters[-1] - 0.5, mismatch[-1] * 10),
            fontsize=8, color=GREEN,
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
        )

        ax.set_xticks(iters)
        ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=8)
        self._style_ax(ax,
            title="Newton-Raphson Convergence",
            xlabel="Iteration Number",
            ylabel="Max Power Mismatch (pu) — log scale"
        )

        if standalone:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight',
                            facecolor=DARK_BG)
                print(f"[Plotter] Convergence plot saved → {save_path}")
            else:
                plt.show()

    # ==================================================================
    # PLOT 2: BUS VOLTAGE PROFILE
    # ==================================================================
    def plot_voltage_profile(self, ax=None, save_path: str = None):
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor(DARK_BG)

        buses  = self.bus_res['Name'].tolist()
        V_vals = self.bus_res['|V| (pu)'].tolist()
        colors = [
            ACCENT1 if v < 0.95 or v > 1.05
            else GREEN if v >= 1.0
            else BLUE
            for v in V_vals
        ]

        bars = ax.bar(buses, V_vals, color=colors, width=0.5,
                      edgecolor='white', linewidth=0.8)

        # Value labels on bars
        for bar, val in zip(bars, V_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom',
                    color=WHITE, fontsize=9, fontweight='bold')

        ax.axhline(y=0.95, color=ORANGE, linewidth=1.5,
                   linestyle='--', label='Lower limit (0.95 pu)')
        ax.axhline(y=1.05, color=YELLOW, linewidth=1.5,
                   linestyle='--', label='Upper limit (1.05 pu)')
        ax.set_ylim(0.90, 1.10)
        ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=8)
        self._style_ax(ax,
            title="Bus Voltage Profile",
            xlabel="Bus",
            ylabel="Voltage Magnitude |V| (pu)"
        )

        if standalone:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight',
                            facecolor=DARK_BG)
                print(f"[Plotter] Voltage profile saved → {save_path}")
            else:
                plt.show()

    # ==================================================================
    # PLOT 3: BUS VOLTAGE ANGLE PROFILE
    # ==================================================================
    def plot_angle_profile(self, ax=None, save_path: str = None):
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor(DARK_BG)

        buses  = self.bus_res['Name'].tolist()
        angles = self.bus_res['θ (deg)'].tolist()

        bars = ax.bar(buses, angles, color=BLUE, width=0.5,
                      edgecolor='white', linewidth=0.8, alpha=0.85)

        for bar, val in zip(bars, angles):
            offset = 0.05 if val >= 0 else -0.3
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f'{val:.3f}°', ha='center', va='bottom',
                    color=WHITE, fontsize=9, fontweight='bold')

        ax.axhline(y=0, color='white', linewidth=0.8, linestyle='-')
        self._style_ax(ax,
            title="Bus Voltage Angle Profile",
            xlabel="Bus",
            ylabel="Voltage Angle θ (degrees)"
        )

        if standalone:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight',
                            facecolor=DARK_BG)
            else:
                plt.show()

    # ==================================================================
    # PLOT 4: LINE POWER FLOWS
    # ==================================================================
    def plot_line_flows(self, ax=None, save_path: str = None):
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor(DARK_BG)

        lines  = self.line_res['From→To'].tolist()
        P_ij   = self.line_res['P_ij (pu)'].tolist()
        Q_ij   = self.line_res['Q_ij (pu)'].tolist()
        x      = np.arange(len(lines))
        w      = 0.35

        ax.bar(x - w/2, P_ij, w, label='P flow (pu)', color=GREEN,
               edgecolor='white', linewidth=0.6, alpha=0.9)
        ax.bar(x + w/2, Q_ij, w, label='Q flow (pu)', color=BLUE,
               edgecolor='white', linewidth=0.6, alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(lines, rotation=30, ha='right', fontsize=8, color=WHITE)
        ax.axhline(y=0, color='white', linewidth=0.5)
        ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=8)
        self._style_ax(ax,
            title="Line Active & Reactive Power Flows",
            xlabel="Transmission Line",
            ylabel="Power Flow (pu)"
        )

        if standalone:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight',
                            facecolor=DARK_BG)
            else:
                plt.show()

    # ==================================================================
    # PLOT 5: LINE LOSSES
    # ==================================================================
    def plot_line_losses(self, ax=None, save_path: str = None):
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor(DARK_BG)

        lines  = self.line_res['From→To'].tolist()
        losses = [abs(v) for v in self.line_res['P_loss (pu)'].tolist()]
        colors = [ACCENT1 if l > 0.01 else ORANGE for l in losses]

        bars = ax.bar(lines, losses, color=colors, width=0.5,
                      edgecolor='white', linewidth=0.7)

        for bar, val in zip(bars, losses):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0002,
                    f'{val:.4f}', ha='center', va='bottom',
                    color=WHITE, fontsize=8)

        ax.set_xticks(range(len(lines)))
        ax.set_xticklabels(lines, rotation=30, ha='right', fontsize=8, color=WHITE)
        total_loss = sum(losses)
        ax.set_title(
            f"Active Power Losses per Line  (Total = {total_loss:.4f} pu)",
            color=WHITE, fontsize=11, fontweight='bold', pad=10
        )
        self._style_ax(ax, xlabel="Transmission Line", ylabel="P Loss (pu)")

        if standalone:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight',
                            facecolor=DARK_BG)
            else:
                plt.show()

    # ==================================================================
    # COMBINED DASHBOARD (all 5 plots)
    # ==================================================================
    def plot_dashboard(self, save_path: str = None):
        """
        Produce a single figure with all Phase 2 plots arranged in a grid.
        This is the main output image to show in your resume / presentation.
        """
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor(DARK_BG)

        gs = gridspec.GridSpec(3, 3, figure=fig,
                               hspace=0.55, wspace=0.40)

        ax1 = fig.add_subplot(gs[0, 0])     # Convergence
        ax2 = fig.add_subplot(gs[0, 1:])    # Voltage profile (wide)
        ax3 = fig.add_subplot(gs[1, 0])     # Angle profile
        ax4 = fig.add_subplot(gs[1, 1:])    # Line flows (wide)
        ax5 = fig.add_subplot(gs[2, :2])    # Line losses
        ax6 = fig.add_subplot(gs[2, 2])     # Summary text box

        self.plot_convergence(ax=ax1)
        self.plot_voltage_profile(ax=ax2)
        self.plot_angle_profile(ax=ax3)
        self.plot_line_flows(ax=ax4)
        self.plot_line_losses(ax=ax5)

        # ── Summary text panel ────────────────────────────────────────
        ax6.set_facecolor(PANEL_BG)
        ax6.axis('off')
        summary_text = (
            "⚡  SYSTEM SUMMARY\n"
            "─────────────────────────\n"
            f"NR Converged  : {self.summary['NR Converged']}\n"
            f"Iterations    : {self.summary['NR Iterations']}\n\n"
            f"P Generation  : {self.summary['Total P Generation (pu)']} pu\n"
            f"Q Generation  : {self.summary['Total Q Generation (pu)']} pu\n\n"
            f"P Load        : {self.summary['Total P Load (pu)']} pu\n"
            f"Q Load        : {self.summary['Total Q Load (pu)']} pu\n\n"
            f"P Losses      : {self.summary['Total P Loss (pu)']} pu\n"
            f"Loss %        : {self.summary['P Loss % of Generation']}%\n\n"
            f"V Violations  : {self.summary['Voltage Violations']}\n"
            f"Line Overloads: {self.summary['Line Overloads']}"
        )
        ax6.text(0.05, 0.95, summary_text,
                 transform=ax6.transAxes, fontsize=9.5,
                 verticalalignment='top', color=WHITE,
                 fontfamily='monospace',
                 bbox=dict(facecolor=ACCENT2, alpha=0.6,
                           boxstyle='round,pad=0.5'))

        fig.suptitle(
            "⚡  Power Grid Load Flow Analysis Dashboard  —  IEEE 5-Bus System",
            color=WHITE, fontsize=15, fontweight='bold', y=1.01
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=DARK_BG)
            print(f"[Plotter] Dashboard saved → {save_path}")
        else:
            plt.show()

        return fig