"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: Phase 3 Plotter (core/phase3_plotter.py)

  Plots produced:
    1. 3-phase waveform comparison — all 4 fault types side-by-side
    2. Fault current magnitude bar chart — per fault type & bus
    3. Voltage sag heatmap — fault bus vs all network buses
    4. Sequence current phasor diagram — |Ia0|, |Ia1|, |Ia2|
    5. Feature distribution violin plots — for ML preview
    6. Combined Phase 3 dashboard
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import pandas as pd
from Core.Waveform_Generator import Waveform
from Core.Fault_Engine import FaultResult


DARK_BG  = '#1A1A2E'
PANEL_BG = '#16213E'
WHITE    = '#ECEFF4'
RED      = '#E74C3C'
GREEN    = '#2ECC71'
BLUE     = '#3498DB'
YELLOW   = '#F9CA24'
ORANGE   = '#E67E22'
PURPLE   = '#9B59B6'
CYAN     = '#1ABC9C'

PHASE_COLORS = {'Va': RED, 'Vb': GREEN, 'Vc': BLUE,
                'Ia': ORANGE, 'Ib': PURPLE, 'Ic': CYAN}


class Phase3Plotter:

    def __init__(self):
        plt.rcParams.update({
            'font.family'     : 'DejaVu Sans',
            'axes.titlecolor' : WHITE,
            'axes.labelcolor' : WHITE,
            'xtick.color'     : WHITE,
            'ytick.color'     : WHITE,
        })

    def _style(self, ax, title='', xlabel='', ylabel=''):
        ax.set_facecolor(PANEL_BG)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_color('#555')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.35, color='#aaa')
        if title:  ax.set_title(title,  color=WHITE, fontsize=10, fontweight='bold', pad=8)
        if xlabel: ax.set_xlabel(xlabel,color=WHITE, fontsize=8)
        if ylabel: ax.set_ylabel(ylabel,color=WHITE, fontsize=8)

    # ==================================================================
    # PLOT 1: WAVEFORM COMPARISON (all 4 fault types + no fault)
    # ==================================================================
    def plot_waveform_grid(self, waveforms: list, save_path: str = None):
        """
        Plot 5 waveforms (No Fault + 4 fault types) in a grid,
        showing Va, Vb, Vc voltages and fault window.
        """
        n_plots = len(waveforms)
        fig, axes = plt.subplots(n_plots, 1, figsize=(16, 3.5 * n_plots))
        fig.patch.set_facecolor(DARK_BG)

        if n_plots == 1:
            axes = [axes]

        for ax, wf in zip(axes, waveforms):
            ax.set_facecolor(PANEL_BG)
            t_ms = wf.t * 1000   # convert to ms

            ax.plot(t_ms, wf.Va, color=RED,   lw=1.4, label='Va', alpha=0.9)
            ax.plot(t_ms, wf.Vb, color=GREEN, lw=1.4, label='Vb', alpha=0.9)
            ax.plot(t_ms, wf.Vc, color=BLUE,  lw=1.4, label='Vc', alpha=0.9)
            ax.plot(t_ms, wf.Ia, color=ORANGE,lw=1.0, label='Ia',
                    alpha=0.7, linestyle='--')

            # Shade fault window
            ax.axvspan(40, 80, color=YELLOW, alpha=0.07, label='Fault window')
            ax.axvline(x=40, color=YELLOW, lw=1.2, linestyle='--', alpha=0.6)
            ax.axvline(x=80, color=YELLOW, lw=1.2, linestyle='--', alpha=0.6)
            ax.axhline(y=0,  color='#555',  lw=0.6)

            ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=7,
                      loc='upper right', ncol=5)
            self._style(ax,
                title=f"Fault Type: {wf.fault_type.replace('_',' ')}  "
                      f"(Bus-{wf.fault_bus})",
                xlabel="Time (ms)", ylabel="Amplitude (pu)")

            for sp in ['bottom', 'left', 'top', 'right']:
                ax.spines[sp].set_color('#555')

        fig.suptitle("3-Phase Waveforms — All Fault Types",
                     color=WHITE, fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=DARK_BG)
            print(f"[Phase3Plotter] Waveform grid saved → {save_path}")
        else:
            plt.show()
        return fig

    # ==================================================================
    # PLOT 2: FAULT CURRENT BAR CHART
    # ==================================================================
    def plot_fault_currents(self, results: list, save_path: str = None):
        """Bar chart of fault current magnitude for each fault type × bus."""
        fault_types = ["Three_Phase", "Line_to_Ground",
                       "Line_to_Line", "Double_Line_to_Ground"]
        n_buses     = 5
        x           = np.arange(n_buses)
        w           = 0.18
        colors      = [RED, GREEN, BLUE, YELLOW]

        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(PANEL_BG)

        for fi, ft in enumerate(fault_types):
            ft_results = [r for r in results if r.fault_type == ft]
            ft_results.sort(key=lambda r: r.fault_bus)
            vals = [r.If_mag for r in ft_results[:n_buses]]
            offset = (fi - 1.5) * w
            bars = ax.bar(x + offset, vals, w, label=ft.replace('_', ' '),
                          color=colors[fi], edgecolor='white',
                          linewidth=0.6, alpha=0.88)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.05,
                        f'{v:.2f}', ha='center', va='bottom',
                        color=WHITE, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([f'Bus-{i+1}' for i in range(n_buses)],
                           color=WHITE)
        ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=8)
        self._style(ax,
            title="Fault Current Magnitude by Fault Type and Bus (pu)",
            xlabel="Fault Bus", ylabel="|If| (pu)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=DARK_BG)
            print(f"[Phase3Plotter] Fault current chart saved → {save_path}")
        plt.close()

    # ==================================================================
    # PLOT 3: VOLTAGE SAG HEATMAP
    # ==================================================================
    def plot_voltage_sag_heatmap(self, results: list,
                                  save_path: str = None):
        """
        Heatmap: rows = fault type, cols = network bus,
        color = voltage sag depth (pu) at that bus.
        """
        fault_types = ["Three_Phase", "Line_to_Ground",
                       "Line_to_Line", "Double_Line_to_Ground"]
        n_buses  = 5
        sag_data = np.zeros((len(fault_types), n_buses))

        for fi, ft in enumerate(fault_types):
            ft_results = [r for r in results
                          if r.fault_type == ft and r.fault_bus == 3]
            if ft_results:
                sag_data[fi] = ft_results[0].voltage_sag[:n_buses]

        fig, ax = plt.subplots(figsize=(11, 5))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(PANEL_BG)

        im = ax.imshow(sag_data, cmap='YlOrRd', aspect='auto',
                       vmin=0, vmax=max(sag_data.max(), 0.01))
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color=WHITE)
        cbar.set_label('Voltage Sag (pu)', color=WHITE)

        ax.set_xticks(range(n_buses))
        ax.set_xticklabels([f'Bus-{i+1}' for i in range(n_buses)], color=WHITE)
        ax.set_yticks(range(len(fault_types)))
        ax.set_yticklabels([ft.replace('_', ' ') for ft in fault_types],
                           color=WHITE, fontsize=9)

        for i in range(len(fault_types)):
            for j in range(n_buses):
                val = sag_data[i, j]
                color = 'black' if val > 0.2 else WHITE
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

        self._style(ax,
            title="Voltage Sag Heatmap — Fault at Bus-3 (pu)",
            xlabel="Affected Bus", ylabel="Fault Type")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=DARK_BG)
            print(f"[Phase3Plotter] Sag heatmap saved → {save_path}")
        plt.close()

    # ==================================================================
    # PLOT 4: SEQUENCE CURRENT COMPARISON
    # ==================================================================
    def plot_sequence_currents(self, results: list, save_path: str = None):
        """
        Bar chart: |Ia0|, |Ia1|, |Ia2| for each fault type at Bus-3.
        This directly shows how sequence networks differ per fault type.
        """
        fault_types = ["Three_Phase", "Line_to_Ground",
                       "Line_to_Line", "Double_Line_to_Ground"]
        labels      = [ft.replace('_', ' ') for ft in fault_types]
        seqs        = {'|Ia1| (pos)': [], '|Ia2| (neg)': [], '|Ia0| (zero)': []}

        for ft in fault_types:
            r = next((x for x in results
                      if x.fault_type == ft and x.fault_bus == 3), None)
            if r:
                seqs['|Ia1| (pos)'].append(abs(r.Ia1))
                seqs['|Ia2| (neg)'].append(abs(r.Ia2))
                seqs['|Ia0| (zero)'].append(abs(r.Ia0))
            else:
                for k in seqs:
                    seqs[k].append(0)

        x  = np.arange(len(fault_types))
        w  = 0.25
        cs = [BLUE, RED, GREEN]

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(DARK_BG)

        for i, (seq_name, vals) in enumerate(seqs.items()):
            bars = ax.bar(x + (i-1)*w, vals, w, label=seq_name,
                          color=cs[i], edgecolor='white',
                          linewidth=0.6, alpha=0.9)
            for bar, v in zip(bars, vals):
                if v > 0.01:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.02,
                            f'{v:.3f}', ha='center', va='bottom',
                            color=WHITE, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=WHITE, fontsize=9)
        ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=9)
        self._style(ax,
            title="Sequence Fault Currents by Fault Type — Bus-3 (pu)",
            xlabel="Fault Type",
            ylabel="Sequence Current Magnitude (pu)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=DARK_BG)
            print(f"[Phase3Plotter] Sequence current chart saved → {save_path}")
        plt.close()

    # ==================================================================
    # PLOT 5: FEATURE DISTRIBUTIONS
    # ==================================================================
    def plot_feature_distributions(self, df: pd.DataFrame,
                                   save_path: str = None):
        """Violin plots of key features grouped by fault class."""
        key_feats = ['Va_sag', 'Vb_sag', 'Vc_sag',
                     'I_max_peak', 'Va_H3', 'Ia_H2',
                     'V0_ratio', 'V2_ratio', 'I_imbalance']
        n = len(key_feats)
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        fig.patch.set_facecolor(DARK_BG)
        axes = axes.flatten()

        classes = df['fault_type'].unique()
        colors  = [RED, GREEN, BLUE, YELLOW, ORANGE]

        for ax, feat in zip(axes, key_feats):
            ax.set_facecolor(PANEL_BG)
            data_by_class = [df[df['fault_type'] == c][feat].values
                             for c in classes]
            vp = ax.violinplot(data_by_class, positions=range(len(classes)),
                               showmedians=True, showmeans=False)
            for i, (body, c) in enumerate(zip(vp['bodies'], colors[:len(classes)])):
                body.set_facecolor(c)
                body.set_alpha(0.7)
            vp['cmedians'].set_colors([WHITE] * len(classes))
            vp['cbars'].set_colors(['#555'] * len(classes))
            vp['cmins'].set_colors(['#555'] * len(classes))
            vp['cmaxes'].set_colors(['#555'] * len(classes))

            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(
                [c.replace('_', '\n') for c in classes],
                color=WHITE, fontsize=6.5)
            self._style(ax, title=feat.replace('_', ' '))

        fig.suptitle("Feature Distributions by Fault Class",
                     color=WHITE, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=DARK_BG)
            print(f"[Phase3Plotter] Feature distributions saved → {save_path}")
        plt.close()