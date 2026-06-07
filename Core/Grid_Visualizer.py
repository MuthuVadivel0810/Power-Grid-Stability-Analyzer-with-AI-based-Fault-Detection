"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: Grid Topology Visualizer (core/grid_visualizer.py)

  What this does:
    - Draws the power network as a graph (buses = nodes, lines = edges)
    - Color-codes buses by type: Slack / PV / PQ
    - Annotates edge weights with line impedance
    - Shows bus voltages as node labels
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


# ── Colour palette ────────────────────────────────────────────────────────────
BUS_COLORS = {
    "Slack": "#E74C3C",   # Red   — reference bus
    "PV"   : "#2ECC71",   # Green — generator bus
    "PQ"   : "#3498DB",   # Blue  — load bus
}


class GridVisualizer:
    """
    Builds and draws a NetworkX graph of the power system.

    Parameters
    ----------
    bus_file  : str  – path to bus_data.csv
    line_file : str  – path to line_data.csv
    """

    def __init__(self, bus_file: str, line_file: str):
        self.bus_data  = pd.read_csv(bus_file)
        self.line_data = pd.read_csv(line_file)
        self.G         = nx.Graph()
        self._build_graph()

    # ------------------------------------------------------------------
    def _build_graph(self):
        """Add nodes (buses) and edges (lines) to the NetworkX graph."""

        # ── Nodes ─────────────────────────────────────────────────────
        for _, bus in self.bus_data.iterrows():
            self.G.add_node(
                int(bus['bus_id']),
                label    = bus['bus_name'],
                bus_type = bus['bus_type'],
                V_mag    = bus['V_mag'],
                P_load   = bus['P_load_pu'],
                Q_load   = bus['Q_load_pu'],
            )

        # ── Edges ─────────────────────────────────────────────────────
        for _, line in self.line_data.iterrows():
            Z_mag = round(np.sqrt(line['R_pu']**2 + line['X_pu']**2), 4)
            self.G.add_edge(
                int(line['from_bus']),
                int(line['to_bus']),
                R      = line['R_pu'],
                X      = line['X_pu'],
                B      = line['B_pu'],
                Z_mag  = Z_mag,
                label  = f"Z={Z_mag:.3f}"
            )

    # ------------------------------------------------------------------
    def plot_topology(self, title: str = "IEEE 5-Bus Power System Topology",
                      save_path: str = None):
        """
        Draw the network topology with bus types colour-coded.

        Parameters
        ----------
        title     : Plot title
        save_path : If given, saves figure to this path instead of showing
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#1A1A2E')
        ax.set_facecolor('#16213E')

        # ── Layout ────────────────────────────────────────────────────
        # Fixed positions for clean IEEE 5-bus layout
        pos = {
            1: (0.0, 0.8),
            2: (0.5, 1.0),
            3: (0.0, 0.2),
            4: (0.5, 0.0),
            5: (1.0, 0.5),
        }

        # ── Node colours based on bus type ────────────────────────────
        node_colors = [
            BUS_COLORS[self.G.nodes[n]['bus_type']]
            for n in self.G.nodes()
        ]

        # ── Node labels: Bus name + Voltage magnitude ─────────────────
        node_labels = {
            n: f"Bus-{n}\n{self.G.nodes[n]['bus_type']}\nV={self.G.nodes[n]['V_mag']:.3f} pu"
            for n in self.G.nodes()
        }

        # ── Edge labels: impedance magnitude ──────────────────────────
        edge_labels = {
            (u, v): f"Z={d['Z_mag']:.3f}"
            for u, v, d in self.G.edges(data=True)
        }

        # ── Draw everything ───────────────────────────────────────────
        nx.draw_networkx_edges(
            self.G, pos, ax=ax,
            edge_color='#A8D8EA', width=2.5, alpha=0.85
        )
        nx.draw_networkx_nodes(
            self.G, pos, ax=ax,
            node_color=node_colors, node_size=2000, alpha=0.95
        )
        nx.draw_networkx_labels(
            self.G, pos, labels=node_labels, ax=ax,
            font_size=8, font_color='white', font_weight='bold'
        )
        nx.draw_networkx_edge_labels(
            self.G, pos, edge_labels=edge_labels, ax=ax,
            font_size=7, font_color='#F9CA24',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#1A1A2E', alpha=0.7)
        )

        # ── Legend ────────────────────────────────────────────────────
        legend_patches = [
            mpatches.Patch(color=c, label=t)
            for t, c in BUS_COLORS.items()
        ]
        ax.legend(
            handles=legend_patches, loc='upper right',
            facecolor='#1A1A2E', edgecolor='white',
            labelcolor='white', fontsize=10
        )

        ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=15)
        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"[GridVisualizer] Topology saved → {save_path}")
        else:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    def plot_ybus_heatmap(self, Ybus: np.ndarray, save_path: str = None):
        """
        Draw a heatmap of |Ybus| to visualize matrix sparsity and magnitude.

        Parameters
        ----------
        Ybus      : complex ndarray from YbusBuilder
        save_path : If given, saves figure here
        """
        n = Ybus.shape[0]
        Ybus_mag = np.abs(Ybus)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#1A1A2E')

        bus_labels = [f"Bus-{i+1}" for i in range(n)]

        for ax, data, title, cmap in zip(
            axes,
            [Ybus_mag, Ybus.real],
            ["|Ybus| — Admittance Magnitudes (pu)", "G matrix — Conductance (pu)"],
            ["YlOrRd", "RdBu"]
        ):
            ax.set_facecolor('#16213E')
            im = ax.imshow(data, cmap=cmap, aspect='auto')
            ax.set_xticks(range(n)); ax.set_xticklabels(bus_labels, color='white', fontsize=9)
            ax.set_yticks(range(n)); ax.set_yticklabels(bus_labels, color='white', fontsize=9)
            ax.set_title(title, color='white', fontsize=11, pad=10)
            plt.colorbar(im, ax=ax)

            # Annotate cells with values
            for i in range(n):
                for j in range(n):
                    val = data[i, j]
                    color = 'white' if val < np.max(data) * 0.6 else 'black'
                    ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                            fontsize=7, color=color)

        fig.suptitle("Ybus Matrix Visualization", color='white',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"[GridVisualizer] Ybus heatmap saved → {save_path}")
        else:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    def get_graph_metrics(self) -> dict:
        """Return basic network analysis metrics."""
        return {
            "n_nodes"         : self.G.number_of_nodes(),
            "n_edges"         : self.G.number_of_edges(),
            "is_connected"    : nx.is_connected(self.G),
            "avg_degree"      : round(np.mean([d for _, d in self.G.degree()]), 2),
            "diameter"        : nx.diameter(self.G),
            "density"         : round(nx.density(self.G), 4),
        }
