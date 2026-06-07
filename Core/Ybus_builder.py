"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: Ybus Matrix Builder (core/ybus_builder.py)
  
  What this does:
    - Reads IEEE bus and line data from CSV files
    - Constructs the Bus Admittance Matrix (Ybus)
    - Ybus is the foundation of ALL power flow calculations
  
  Theory:
    For a transmission line between bus i and bus j:
      - Series admittance:  y_ij = 1 / (R_ij + jX_ij)
      - Shunt admittance:   y_sh = jB_ij/2  (at each end)
    
    Ybus elements:
      - Off-diagonal: Y_ij = -y_ij          (mutual admittance)
      - Diagonal:     Y_ii = sum(y_ij) + sum(y_sh_i)  (self admittance)
=============================================================
"""

import numpy as np
import pandas as pd


class YbusBuilder:
    """
    Constructs the Bus Admittance Matrix (Ybus) for a power system.

    Attributes
    ----------
    n_buses     : int            - Total number of buses
    bus_data    : pd.DataFrame   - Bus data loaded from CSV
    line_data   : pd.DataFrame   - Line data loaded from CSV
    Ybus        : np.ndarray     - Complex Ybus matrix (n x n)
    """

    def __init__(self, bus_file: str, line_file: str):
        """
        Parameters
        ----------
        bus_file  : path to bus_data.csv
        line_file : path to line_data.csv
        """
        self.bus_data  = pd.read_csv(bus_file)
        self.line_data = pd.read_csv(line_file)
        self.n_buses   = len(self.bus_data)
        self.Ybus      = np.zeros((self.n_buses, self.n_buses), dtype=complex)
        print(f"[YbusBuilder] Loaded {self.n_buses} buses and {len(self.line_data)} lines.")

    # ------------------------------------------------------------------
    def build(self) -> np.ndarray:
        """
        Build and return the Ybus matrix.

        Algorithm:
          For each transmission line (from_bus i → to_bus j):
            1. Calculate series admittance: y_series = 1 / (R + jX)
            2. Calculate shunt admittance:  y_shunt  = jB/2
            3. Update Ybus:
               - Y[i,i] += y_series + y_shunt   (diagonal of from-bus)
               - Y[j,j] += y_series + y_shunt   (diagonal of to-bus)
               - Y[i,j] -= y_series             (off-diagonal)
               - Y[j,i] -= y_series             (off-diagonal, symmetric)
        """
        self.Ybus = np.zeros((self.n_buses, self.n_buses), dtype=complex)

        for _, line in self.line_data.iterrows():
            # Bus indices (0-based)
            i = int(line['from_bus']) - 1
            j = int(line['to_bus'])   - 1

            R   = line['R_pu']
            X   = line['X_pu']
            B   = line['B_pu']
            tap = line['tap_ratio']

            # --- Series admittance of the line ---
            z_series = complex(R, X)           # series impedance
            y_series = 1.0 / z_series          # series admittance

            # --- Shunt susceptance (line charging, split at both ends) ---
            y_shunt = complex(0, B / 2.0)

            # --- Handle transformer tap ratio ---
            # When tap != 1.0, the line is a transformer branch
            if tap != 1.0:
                # Off-nominal tap transformer π-model
                self.Ybus[i, i] += y_series / (tap ** 2)
                self.Ybus[j, j] += y_series
                self.Ybus[i, j] -= y_series / tap
                self.Ybus[j, i] -= y_series / tap
            else:
                # Standard transmission line π-model
                self.Ybus[i, i] += y_series + y_shunt
                self.Ybus[j, j] += y_series + y_shunt
                self.Ybus[i, j] -= y_series
                self.Ybus[j, i] -= y_series          # Ybus is symmetric for passive networks

        print("[YbusBuilder] Ybus matrix built successfully.")
        return self.Ybus

    # ------------------------------------------------------------------
    def get_conductance_matrix(self) -> np.ndarray:
        """Return G = Real part of Ybus"""
        return self.Ybus.real

    def get_susceptance_matrix(self) -> np.ndarray:
        """Return B = Imaginary part of Ybus"""
        return self.Ybus.imag

    # ------------------------------------------------------------------
    def print_ybus(self):
        """Pretty-print Ybus with G + jB format for each element."""
        print("\n" + "=" * 65)
        print("  BUS ADMITTANCE MATRIX (Ybus) — Values in per-unit")
        print("=" * 65)
        header = "     " + "".join(f"{'Bus-'+str(k+1):>18}" for k in range(self.n_buses))
        print(header)
        print("-" * 65)
        for i in range(self.n_buses):
            row_str = f"Bus-{i+1} "
            for j in range(self.n_buses):
                g = self.Ybus[i, j].real
                b = self.Ybus[i, j].imag
                sign = '+' if b >= 0 else '-'
                row_str += f"  {g:6.4f}{sign}j{abs(b):6.4f}"
            print(row_str)
        print("=" * 65)

    # ------------------------------------------------------------------
    def get_summary(self) -> dict:
        """Return a dict of key Ybus metrics for display/testing."""
        diag = np.diag(self.Ybus)
        return {
            "n_buses"          : self.n_buses,
            "n_lines"          : len(self.line_data),
            "is_symmetric"     : np.allclose(self.Ybus, self.Ybus.T, atol=1e-10),
            "diagonal_G_pu"    : self.Ybus.real.diagonal().tolist(),
            "diagonal_B_pu"    : self.Ybus.imag.diagonal().tolist(),
            "sparsity_percent" : round(
                100.0 * np.sum(np.abs(self.Ybus) < 1e-10) / self.Ybus.size, 2
            ),
        }
