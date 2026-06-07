"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Phase 3 Runner : phase3_main.py

  Pipeline:
    1. Re-run NR solver  → get post-load-flow voltages
    2. Build FaultEngine → Z1, Z2, Z0 sequence networks
    3. Compute all 4 fault types at every bus (Rf = 0)
    4. Print fault reports for Bus-3 (mid-grid bus)
    5. Generate one waveform per fault type for plotting
    6. Build full labeled ML dataset  (1000 samples)
    7. Save dataset CSV  → ml/fault_dataset.csv
    8. Generate all Phase 3 plots + dashboard

  Run with:  python phase3_main.py
=============================================================
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from Core.Ybus_builder      import YbusBuilder
from Core.NR_solver         import NRSolver
from Core.Fault_Engine      import FaultEngine, FAULT_TYPES
from Core.Waveform_Generator import WaveformGenerator
from Core.Dataset_Generator  import FeatureExtractor, DatasetGenerator
from Core.Phase3_Plotter     import Phase3Plotter

# ── Paths ─────────────────────────────────────────────────────────────────────
BUS_FILE  = "data/bus_data.csv"
LINE_FILE = "data/line_data.csv"
OUT_DIR   = "outputs/phase3"
ML_DIR    = "ml"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ML_DIR,  exist_ok=True)


def main():
    print("\n" + "═"*68)
    print("  ⚡  POWER GRID STABILITY ANALYZER  —  Phase 3")
    print("      Fault Injection Engine + Dataset Generation")
    print("═"*68)

    # ══════════════════════════════════════════════════════════════════
    # STEP 1 — Re-run NR solver to get pre-fault voltages
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 1] Building Ybus & running NR load flow...")
    builder = YbusBuilder(BUS_FILE, LINE_FILE)
    Ybus    = builder.build()

    solver    = NRSolver(BUS_FILE, LINE_FILE, Ybus, tolerance=1e-6)
    nr_result = solver.solve()

    # Build complex pre-fault voltage phasors:  V∠θ  for each bus
    V_prefault = np.array([
        nr_result['V'][i] * np.exp(1j * nr_result['theta_rad'][i])
        for i in range(len(nr_result['V']))
    ])

    print(f"\n  Pre-fault bus voltages (from NR):")
    for i, v in enumerate(V_prefault):
        print(f"    Bus-{i+1}: {abs(v):.5f} pu  ∠{np.rad2deg(np.angle(v)):+.3f}°")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2 — Build Fault Engine
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 2] Initializing Fault Engine...")
    fe = FaultEngine(Ybus, V_prefault, Z0_factor=3.0)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3 — Run all 4 fault types at every bus (bolted, Rf = 0)
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 3] Running all fault types at all buses (bolted fault)...")
    fault_names = list(FAULT_TYPES.values())[1:]   # skip "No_Fault"
    all_results = []

    for ft in fault_names:
        for bus_idx in range(fe.n):
            r = fe.run_fault(ft, bus_idx, Rf=0.0)
            all_results.append(r)

    print(f"  Computed {len(all_results)} fault scenarios  ✓")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4 — Detailed fault report for Bus-3 (all fault types)
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 4] Detailed fault analysis — Fault Bus = Bus-3")
    bus3_results = [r for r in all_results if r.fault_bus == 3]
    for r in bus3_results:
        fe.print_fault_report(r)

    # ══════════════════════════════════════════════════════════════════
    # STEP 5 — Fault current summary table
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 5] Fault current summary table")
    print(f"\n  {'Fault Type':<30} {'Bus-1':>8} {'Bus-2':>8} "
          f"{'Bus-3':>8} {'Bus-4':>8} {'Bus-5':>8}")
    print("  " + "-"*66)
    for ft in fault_names:
        row = f"  {ft:<30}"
        for bus_idx in range(fe.n):
            r = next(x for x in all_results
                     if x.fault_type == ft and x.fault_bus == bus_idx+1)
            row += f" {r.If_mag:>8.4f}"
        print(row)
    print("  " + "-"*66)
    print(f"  Units: per-unit  |  Base: 100 MVA\n")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6 — Generate waveforms for all fault types at Bus-3
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 6] Generating 3-phase waveforms...")
    wg = WaveformGenerator(
        fs=6400, duration=0.1,
        fault_start=0.04, fault_dur=0.04,
        freq=50.0, noise_std=0.005
    )

    waveforms = []
    # No-fault first
    wf_nf = wg.generate_no_fault(bus_idx=2,
                                  V_mag=abs(V_prefault[2]))
    waveforms.append(wf_nf)

    # One waveform per fault type (fault at Bus-3)
    for ft in fault_names:
        r  = next(x for x in all_results
                  if x.fault_type == ft and x.fault_bus == 3)
        wf = wg.generate_fault_waveform(r)
        waveforms.append(wf)
        print(f"  {ft:<30} → waveform generated  "
              f"(If = {r.If_mag:.4f} pu)")

    print(f"\n  Total waveforms: {len(waveforms)}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 7 — Generate ML dataset
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 7] Generating ML training dataset...")
    fex  = FeatureExtractor(fs=6400, freq=50.0)
    dgen = DatasetGenerator(fe, wg, fex, n_buses=fe.n)

    # 40 Rf values → 200 no-fault + 800 fault = 1000 samples
    Rf_sweep = np.concatenate([[0.0], np.logspace(-3, -1, 39)])
    dataset  = dgen.generate(Rf_values=Rf_sweep, seed=42)

    dataset_path = f"{ML_DIR}/fault_dataset.csv"
    dgen.save_dataset(dataset, dataset_path)

    # ══════════════════════════════════════════════════════════════════
    # STEP 8 — Save fault results summary CSV
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 8] Saving fault results summary CSV...")
    rows = []
    for r in all_results:
        rows.append({
            "fault_type"   : r.fault_type,
            "fault_bus"    : r.fault_bus,
            "Rf_pu"        : r.fault_resistance,
            "Vf_mag"       : round(abs(r.Vf),  5),
            "|Z1| pu"      : round(abs(r.Z1),  5),
            "|Z2| pu"      : round(abs(r.Z2),  5),
            "|Z0| pu"      : round(abs(r.Z0),  5),
            "|Ia1| pu"     : round(abs(r.Ia1), 5),
            "|Ia2| pu"     : round(abs(r.Ia2), 5),
            "|Ia0| pu"     : round(abs(r.Ia0), 5),
            "|If_total| pu": round(r.If_mag,   5),
            "V_sag_fbus"   : round(r.voltage_sag[r.fault_bus-1], 5),
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(f"{OUT_DIR}/fault_results_summary.csv", index=False)
    print(f"  Saved: {OUT_DIR}/fault_results_summary.csv")

    # ══════════════════════════════════════════════════════════════════
    # STEP 9 — Generate all plots
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 9] Generating Phase 3 plots...")
    plotter = Phase3Plotter()

    plotter.plot_waveform_grid(
        waveforms,
        save_path=f"{OUT_DIR}/waveforms_all_faults.png")

    plotter.plot_fault_currents(
        all_results,
        save_path=f"{OUT_DIR}/fault_currents.png")

    plotter.plot_voltage_sag_heatmap(
        all_results,
        save_path=f"{OUT_DIR}/voltage_sag_heatmap.png")

    plotter.plot_sequence_currents(
        all_results,
        save_path=f"{OUT_DIR}/sequence_currents.png")

    plotter.plot_feature_distributions(
        dataset,
        save_path=f"{OUT_DIR}/feature_distributions.png")

    # ══════════════════════════════════════════════════════════════════
    # STEP 10 — Print dataset statistics
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 10] Dataset statistics for ML phase:")
    feature_cols = [c for c in dataset.columns
                    if c not in ('fault_type', 'label', 'Rf_pu', 'fault_bus')]
    print(f"\n  Total samples   : {len(dataset)}")
    print(f"  Feature count   : {len(feature_cols)}")
    print(f"  Target classes  : {dataset['label'].nunique()}")
    print(f"\n  Feature statistics (fault window):")
    print(dataset[feature_cols[:8]].describe().round(4).to_string())

    print("\n" + "═"*68)
    print("  ✅  Phase 3 COMPLETE")
    print(f"\n  Outputs saved to:")
    print(f"    {OUT_DIR}/waveforms_all_faults.png")
    print(f"    {OUT_DIR}/fault_currents.png")
    print(f"    {OUT_DIR}/voltage_sag_heatmap.png")
    print(f"    {OUT_DIR}/sequence_currents.png")
    print(f"    {OUT_DIR}/feature_distributions.png")
    print(f"    {OUT_DIR}/fault_results_summary.csv")
    print(f"    {ML_DIR}/fault_dataset.csv  ← Ready for ML (Phase 4)")
    print(f"\n  Next → Phase 4: ML Fault Classifier")
    print("═"*68 + "\n")


if __name__ == "__main__":
    main()