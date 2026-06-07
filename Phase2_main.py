"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Phase 2 Runner: phase2_main.py

  What this runs:
    1. Build Ybus  (reuses Phase 1 YbusBuilder)
    2. Run Newton-Raphson load flow solver
    3. Analyze results  (bus voltages, line flows, losses)
    4. Print full report to console
    5. Export results to CSV files
    6. Generate all 5 plots + combined dashboard
    7. Save everything to outputs/phase2/

  Run with:  python phase2_main.py
=============================================================
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from Core.Ybus_builder    import YbusBuilder
from Core.NR_solver       import NRSolver
from Core.Result_Analyzer import ResultsAnalyzer
from Core.Phase2_Plotter  import Phase2Plotter


# ── Paths ─────────────────────────────────────────────────────────────────────
BUS_FILE  = "data/bus_data.csv"
LINE_FILE = "data/line_data.csv"
OUT_DIR   = "outputs/phase2"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("\n" + "=" * 70)
    print("  ⚡  POWER GRID STABILITY ANALYZER  —  Phase 2")
    print("      Newton-Raphson Load Flow Solver")
    print("=" * 70)

    # ── STEP 1: Build Ybus ────────────────────────────────────────────
    print("\n[STEP 1] Building Ybus matrix...")
    builder = YbusBuilder(BUS_FILE, LINE_FILE)
    Ybus    = builder.build()
    print("  Ybus built ✓")

    # ── STEP 2: Run NR Solver ─────────────────────────────────────────
    print("\n[STEP 2] Running Newton-Raphson Load Flow...")
    solver    = NRSolver(BUS_FILE, LINE_FILE, Ybus,
                         tolerance=1e-6, max_iter=50)
    nr_result = solver.solve()

    if not nr_result['converged']:
        print("\n❌ Load flow did not converge. Check your data.")
        sys.exit(1)

    # ── STEP 3: Print Convergence Summary ────────────────────────────
    print(f"\n[STEP 3] Convergence Summary:")
    print(f"  Converged in  : {nr_result['iterations']} iterations")
    print(f"  Final mismatch: {nr_result['mismatch_history'][-1]:.4e} pu")

    print(f"\n  Final Bus Voltages & Angles:")
    print(f"  {'Bus':<6} {'|V| (pu)':<12} {'θ (deg)':<12}")
    print(f"  {'-'*30}")
    for i in range(len(nr_result['V'])):
        print(f"  Bus-{i+1}  "
              f"{nr_result['V'][i]:<12.6f}"
              f"{nr_result['theta_deg'][i]:<12.6f}")

    # ── STEP 4: Analyze Results ───────────────────────────────────────
    print("\n[STEP 4] Analyzing load flow results...")
    bus_data  = pd.read_csv(BUS_FILE)
    line_data = pd.read_csv(LINE_FILE)
    analyzer  = ResultsAnalyzer(bus_data, line_data, Ybus, nr_result)

    bus_results  = analyzer.get_bus_results()
    line_results = analyzer.get_line_flows()
    summary      = analyzer.get_system_summary()

    analyzer.print_report()

    # ── STEP 5: Export results to CSV ────────────────────────────────
    print("\n[STEP 5] Exporting results to CSV...")
    bus_results.to_csv(f"{OUT_DIR}/bus_results.csv",  index=False)
    line_results.to_csv(f"{OUT_DIR}/line_results.csv", index=False)

    # Convergence history
    conv_df = pd.DataFrame({
        'iteration'    : list(range(1, len(nr_result['mismatch_history']) + 1)),
        'max_mismatch' : nr_result['mismatch_history']
    })
    conv_df.to_csv(f"{OUT_DIR}/convergence_history.csv", index=False)

    print(f"  Saved: {OUT_DIR}/bus_results.csv")
    print(f"  Saved: {OUT_DIR}/line_results.csv")
    print(f"  Saved: {OUT_DIR}/convergence_history.csv")

    # ── STEP 6: Generate all plots ────────────────────────────────────
    print("\n[STEP 6] Generating plots...")
    plotter = Phase2Plotter(bus_results, line_results, nr_result, summary)

    plotter.plot_convergence(
        save_path=f"{OUT_DIR}/plot_convergence.png")
    plotter.plot_voltage_profile(
        save_path=f"{OUT_DIR}/plot_voltage_profile.png")
    plotter.plot_angle_profile(
        save_path=f"{OUT_DIR}/plot_angle_profile.png")
    plotter.plot_line_flows(
        save_path=f"{OUT_DIR}/plot_line_flows.png")
    plotter.plot_line_losses(
        save_path=f"{OUT_DIR}/plot_line_losses.png")

    print("\n[STEP 7] Generating full dashboard...")
    plotter.plot_dashboard(
        save_path=f"{OUT_DIR}/dashboard_phase2.png")

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✅  Phase 2 COMPLETE")
    print(f"  NR converged in {nr_result['iterations']} iterations")
    print(f"  Total P Loss : {summary['Total P Loss (pu)']} pu  "
          f"({summary['P Loss % of Generation']}% of generation)")
    print(f"  Voltage violations : {summary['Voltage Violations']}")
    print(f"  Line overloads     : {summary['Line Overloads']}")
    print(f"\n  All outputs saved to → {OUT_DIR}/")
    print("  Next → Phase 3: Fault Injection Engine")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()