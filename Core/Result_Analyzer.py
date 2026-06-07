"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: Results Analyzer (core/results_analyzer.py)

  What this does:
    - Takes NR solver output (V, θ) and computes:
        1. Line power flows  (P_ij, Q_ij in both directions)
        2. Line losses       (ΔP_loss, ΔQ_loss per line)
        3. Bus voltage check (flag buses with V < 0.95 or V > 1.05)
        4. Generator output  (how much the slack & PV buses supply)
        5. System totals     (total generation, load, losses)
    - Produces all results as DataFrames (easy to export/display)
=============================================================
"""

import numpy as np
import pandas as pd


class ResultsAnalyzer:
    """
    Post-processes Newton-Raphson load flow results.

    Parameters
    ----------
    bus_data  : pd.DataFrame – loaded from bus_data.csv
    line_data : pd.DataFrame – loaded from line_data.csv
    Ybus      : np.ndarray   – complex Ybus matrix
    nr_result : dict         – output dictionary from NRSolver.solve()
    """

    V_MIN = 0.95    # lower voltage limit (pu)
    V_MAX = 1.05    # upper voltage limit (pu)

    def __init__(self, bus_data: pd.DataFrame, line_data: pd.DataFrame,
                 Ybus: np.ndarray, nr_result: dict):
        self.bus_data  = bus_data
        self.line_data = line_data
        self.Ybus      = Ybus
        self.G         = Ybus.real
        self.B         = Ybus.imag
        self.nr        = nr_result

        self.V         = nr_result['V']
        self.theta     = nr_result['theta_rad']
        self.n         = len(bus_data)

    # ==================================================================
    # 1. BUS VOLTAGE RESULTS TABLE
    # ==================================================================
    def get_bus_results(self) -> pd.DataFrame:
        """
        Returns a DataFrame with per-bus voltage summary.

        Columns: Bus, Type, |V| pu, θ°, P_gen, Q_gen, P_load, Q_load, Status
        """
        rows = []
        P_calc = self.nr['P_calc']
        Q_calc = self.nr['Q_calc']

        for i, row in self.bus_data.iterrows():
            V_i   = self.V[i]
            th_i  = np.rad2deg(self.theta[i])

            # Generator output: P_gen = P_calc + P_load (from power balance)
            P_gen = round(P_calc[i] + row['P_load_pu'], 4)
            Q_gen = round(Q_calc[i] + row['Q_load_pu'], 4)

            # Voltage violation flag
            if V_i < self.V_MIN:
                status = "⚠ Low Voltage"
            elif V_i > self.V_MAX:
                status = "⚠ High Voltage"
            else:
                status = "✓ Normal"

            rows.append({
                "Bus"       : int(row['bus_id']),
                "Name"      : row['bus_name'],
                "Type"      : row['bus_type'],
                "|V| (pu)"  : round(V_i,  5),
                "θ (deg)"   : round(th_i, 4),
                "P_gen (pu)": round(P_gen, 4) if P_gen > 0.001 else 0.0,
                "Q_gen (pu)": round(Q_gen, 4),
                "P_load(pu)": round(row['P_load_pu'], 4),
                "Q_load(pu)": round(row['Q_load_pu'], 4),
                "Status"    : status,
            })

        return pd.DataFrame(rows)

    # ==================================================================
    # 2. LINE FLOW RESULTS TABLE
    # ==================================================================
    def get_line_flows(self) -> pd.DataFrame:
        """
        Compute power flow on each transmission line.

        For a line from bus i to bus j with series admittance y_s = g+jb
        and shunt y_sh = 0 + jb_sh:

          I_ij = y_s(V_i - V_j) + y_sh * V_i
          S_ij = V_i * I_ij*  = P_ij + jQ_ij   (complex power)
          S_ji = V_j * I_ji*  = P_ji + jQ_ji   (reverse direction)
          Loss = S_ij + S_ji
        """
        rows = []

        for _, line in self.line_data.iterrows():
            i = int(line['from_bus']) - 1
            j = int(line['to_bus'])   - 1

            R   = line['R_pu']
            X   = line['X_pu']
            B   = line['B_pu']
            tap = line['tap_ratio']

            # Series admittance
            z_s   = complex(R, X)
            y_s   = 1.0 / z_s
            y_sh  = complex(0, B / 2.0)

            # Voltage phasors
            Vi = self.V[i] * np.exp(1j * self.theta[i])
            Vj = self.V[j] * np.exp(1j * self.theta[j])

            if tap != 1.0:
                # Transformer branch
                I_ij = (y_s / tap**2) * Vi - (y_s / tap) * Vj
                I_ji = -(y_s / tap) * Vi + y_s * Vj
            else:
                # Standard line
                I_ij = (y_s + y_sh) * Vi - y_s * Vj
                I_ji = (y_s + y_sh) * Vj - y_s * Vi

            # Complex power flows
            S_ij = Vi * np.conj(I_ij)
            S_ji = Vj * np.conj(I_ji)

            P_ij = round(S_ij.real, 5)
            Q_ij = round(S_ij.imag, 5)
            P_ji = round(S_ji.real, 5)
            Q_ji = round(S_ji.imag, 5)

            # Losses on this line
            P_loss = round(P_ij + P_ji, 6)
            Q_loss = round(Q_ij + Q_ji, 6)

            # Current magnitude (for loading check)
            I_mag   = abs(I_ij)
            rating  = line['rating_MVA']   # in MVA base
            loading = round(abs(S_ij) / (rating / 100) * 100, 2)   # % of rating

            rows.append({
                "Line"           : int(line['line_id']),
                "From→To"        : f"Bus{i+1}→Bus{j+1}",
                "P_ij (pu)"      : P_ij,
                "Q_ij (pu)"      : Q_ij,
                "P_ji (pu)"      : P_ji,
                "Q_ji (pu)"      : Q_ji,
                "P_loss (pu)"    : P_loss,
                "Q_loss (pu)"    : Q_loss,
                "|I| (pu)"       : round(I_mag, 5),
                "Loading (%)"    : loading,
                "Status"         : "⚠ Overload" if loading > 100 else "✓ OK",
            })

        return pd.DataFrame(rows)

    # ==================================================================
    # 3. SYSTEM SUMMARY
    # ==================================================================
    def get_system_summary(self) -> dict:
        """
        Total generation, load, and losses across the system.
        """
        bus_res  = self.get_bus_results()
        line_res = self.get_line_flows()

        total_P_gen  = bus_res['P_gen (pu)'].sum()
        total_Q_gen  = bus_res['Q_gen (pu)'].sum()
        total_P_load = bus_res['P_load(pu)'].sum()
        total_Q_load = bus_res['Q_load(pu)'].sum()
        total_P_loss = line_res['P_loss (pu)'].sum()
        total_Q_loss = line_res['Q_loss (pu)'].sum()

        n_violations = bus_res[bus_res['Status'] != '✓ Normal'].shape[0]
        n_overloads  = line_res[line_res['Status'] == '⚠ Overload'].shape[0]

        return {
            "Total P Generation (pu)" : round(total_P_gen,  4),
            "Total Q Generation (pu)" : round(total_Q_gen,  4),
            "Total P Load (pu)"       : round(total_P_load, 4),
            "Total Q Load (pu)"       : round(total_Q_load, 4),
            "Total P Loss (pu)"       : round(total_P_loss, 5),
            "Total Q Loss (pu)"       : round(total_Q_loss, 5),
            "P Loss % of Generation"  : round(abs(total_P_loss / total_P_gen) * 100, 3)
                                        if total_P_gen != 0 else 0,
            "Voltage Violations"      : n_violations,
            "Line Overloads"          : n_overloads,
            "NR Converged"            : self.nr['converged'],
            "NR Iterations"           : self.nr['iterations'],
        }

    # ==================================================================
    # 4. PRINT FULL REPORT
    # ==================================================================
    def print_report(self):
        """Pretty-print the complete load flow report to console."""
        print("\n" + "=" * 70)
        print("  LOAD FLOW RESULTS REPORT")
        print("=" * 70)

        # Bus results
        print("\n  ── BUS VOLTAGE RESULTS ──────────────────────────────────────")
        bus_df = self.get_bus_results()
        print(bus_df.to_string(index=False))

        # Line flows
        print("\n  ── LINE POWER FLOWS ─────────────────────────────────────────")
        line_df = self.get_line_flows()
        print(line_df.to_string(index=False))

        # System summary
        print("\n  ── SYSTEM SUMMARY ───────────────────────────────────────────")
        summary = self.get_system_summary()
        for k, v in summary.items():
            print(f"    {k:<35}: {v}")

        print("=" * 70)