"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Phase 1 Runner: phase1_main.py

  What this runs:
    1. Load IEEE 5-bus data
    2. Build Ybus matrix
    3. Print Ybus with G + jB format
    4. Show key metrics (symmetry, sparsity, diagonal values)
    5. Plot grid topology (NetworkX graph)
    6. Plot Ybus heatmap
    7. Save all outputs to outputs/ folder

  Run with:  python phase1_main.py
=============================================================
"""

import os
import sys
import numpy as np

# ── Make sure core/ is importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from Core.Ybus_builder  import YbusBuilder
from Core.Grid_Visualizer import GridVisualizer


# ── File paths ────────────────────────────────────────────────────────────────
BUS_FILE  = "data/bus_data.csv"
LINE_FILE = "data/line_data.csv"
OUT_DIR   = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("\n" + "=" * 65)
    print("  ⚡  POWER GRID STABILITY ANALYZER  —  Phase 1")
    print("      Ybus Matrix Construction + Grid Topology")
    print("=" * 65)

    # ── STEP 1: Build Ybus ────────────────────────────────────────────
    print("\n[STEP 1] Building Bus Admittance Matrix (Ybus)...")
    builder = YbusBuilder(BUS_FILE, LINE_FILE)
    Ybus    = builder.build()

    # ── STEP 2: Print Ybus ────────────────────────────────────────────
    print("\n[STEP 2] Ybus Matrix (G + jB format):")
    builder.print_ybus()

    # ── STEP 3: Summary metrics ───────────────────────────────────────
    print("\n[STEP 3] Ybus Summary Metrics:")
    summary = builder.get_summary()
    print(f"  • Number of buses      : {summary['n_buses']}")
    print(f"  • Number of lines      : {summary['n_lines']}")
    print(f"  • Matrix symmetric?    : {summary['is_symmetric']}  ← Must be True for passive network")
    print(f"  • Sparsity             : {summary['sparsity_percent']}%")
    print("\n  Diagonal Conductance (G_ii) — Self-conductance of each bus:")
    for i, g in enumerate(summary['diagonal_G_pu']):
        print(f"    Bus-{i+1}: G = {g:.6f} pu")
    print("\n  Diagonal Susceptance (B_ii) — Self-susceptance of each bus:")
    for i, b in enumerate(summary['diagonal_B_pu']):
        print(f"    Bus-{i+1}: B = {b:.6f} pu")

    # ── STEP 4: Validate Ybus ─────────────────────────────────────────
    print("\n[STEP 4] Validation Checks:")
    G = builder.get_conductance_matrix()
    B = builder.get_susceptance_matrix()

    # Check 1: Row sums of G should be near zero (KCL)
    row_sums_G = np.sum(G, axis=1)
    print("  KCL Check — Row sums of G matrix (should be ≈ 0 if no shunts):")
    for i, s in enumerate(row_sums_G):
        status = "✓" if abs(s) < 0.1 else "✗"
        print(f"    Bus-{i+1}: {s:.6f}  {status}")

    # Check 2: Ybus should be symmetric
    sym_err = np.max(np.abs(Ybus - Ybus.T))
    print(f"\n  Symmetry error (max|Ybus - Ybus.T|) = {sym_err:.2e}  {'✓' if sym_err < 1e-10 else '✗'}")

    # Check 3: Diagonal elements should have positive real part
    diag_G = np.diag(G)
    all_pos = np.all(diag_G > 0)
    print(f"  All diagonal conductances > 0?       {'✓ Yes' if all_pos else '✗ No'}")

    # ── STEP 5: Save Ybus to CSV ──────────────────────────────────────
    print("\n[STEP 5] Saving Ybus to CSV...")
    import pandas as pd
    n = Ybus.shape[0]
    labels = [f"Bus-{i+1}" for i in range(n)]

    ybus_df_real = pd.DataFrame(G, index=labels, columns=labels)
    ybus_df_imag = pd.DataFrame(B, index=labels, columns=labels)
    ybus_df_real.to_csv(f"{OUT_DIR}/ybus_conductance_G.csv")
    ybus_df_imag.to_csv(f"{OUT_DIR}/ybus_susceptance_B.csv")
    print(f"  Saved: {OUT_DIR}/ybus_conductance_G.csv")
    print(f"  Saved: {OUT_DIR}/ybus_susceptance_B.csv")

    # ── STEP 6: Visualize topology ────────────────────────────────────
    print("\n[STEP 6] Generating Grid Topology Plot...")
    viz = GridVisualizer(BUS_FILE, LINE_FILE)

    metrics = viz.get_graph_metrics()
    print(f"  Network metrics:")
    print(f"    • Connected graph : {metrics['is_connected']}")
    print(f"    • Average degree  : {metrics['avg_degree']}")
    print(f"    • Diameter        : {metrics['diameter']} hops")
    print(f"    • Density         : {metrics['density']}")

    viz.plot_topology(
        title="IEEE 5-Bus Power System — Grid Topology",
        save_path=f"{OUT_DIR}/grid_topology.png"
    )

    # ── STEP 7: Ybus heatmap ──────────────────────────────────────────
    print("\n[STEP 7] Generating Ybus Heatmap...")
    viz.plot_ybus_heatmap(
        Ybus,
        save_path=f"{OUT_DIR}/ybus_heatmap.png"
    )

    print("\n" + "=" * 65)
    print("  ✅  Phase 1 COMPLETE — All outputs saved to outputs/")
    print("  Next → Phase 2: Newton-Raphson Load Flow Solver")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()