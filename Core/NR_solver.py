"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: Newton-Raphson Load Flow Solver (core/nr_solver.py)

  What this does:
    - Solves the nonlinear power flow equations iteratively
    - Finds bus voltages (magnitude & angle) at every bus
    - Uses the Newton-Raphson method (same algorithm used in
      industry tools like PSS/E, PowerWorld, MATPOWER)

  Theory — Power Flow Equations:
    At every bus i:
      P_i = |V_i| * Σ_j |V_j| (G_ij cos θ_ij + B_ij sin θ_ij)
      Q_i = |V_i| * Σ_j |V_j| (G_ij sin θ_ij - B_ij cos θ_ij)
    where θ_ij = θ_i - θ_j

  Newton-Raphson Iteration:
    [ΔP]   [H  N] [Δθ     ]
    [ΔQ] = [J  L] [Δ|V|/|V|]

    Solve for corrections, update θ and |V|, repeat until
    max(|ΔP|, |ΔQ|) < tolerance (convergence)

  Bus Types:
    Slack (Vθ): V and θ known  → no equations
    PV    (PV): P and V known  → ΔP equation only
    PQ    (PQ): P and Q known  → ΔP and ΔQ equations
=============================================================
"""

import numpy as np
import pandas as pd
from typing import Tuple


class NRSolver:
    """
    Newton-Raphson Power Flow Solver.

    Parameters
    ----------
    bus_file  : str  – path to bus_data.csv
    line_file : str  – path to line_data.csv
    Ybus      : np.ndarray – complex Ybus matrix from YbusBuilder
    tolerance : float – convergence criterion (default 1e-6 pu)
    max_iter  : int   – maximum Newton-Raphson iterations (default 50)
    """

    def __init__(self, bus_file: str, line_file: str,
                 Ybus: np.ndarray,
                 tolerance: float = 1e-6,
                 max_iter:  int   = 50):

        self.bus_data  = pd.read_csv(bus_file)
        self.line_data = pd.read_csv(line_file)
        self.Ybus      = Ybus
        self.tolerance = tolerance
        self.max_iter  = max_iter
        self.n         = len(self.bus_data)      # total number of buses

        # ── Ybus components ───────────────────────────────────────────
        self.G = Ybus.real                        # Conductance matrix
        self.B = Ybus.imag                        # Susceptance matrix

        # ── Bus type classification ───────────────────────────────────
        self.slack_buses = self.bus_data.index[
            self.bus_data['bus_type'] == 'Slack'].tolist()
        self.pv_buses    = self.bus_data.index[
            self.bus_data['bus_type'] == 'PV'].tolist()
        self.pq_buses    = self.bus_data.index[
            self.bus_data['bus_type'] == 'PQ'].tolist()

        # Non-slack buses need ΔP equations
        self.pq_pv_buses = self.pv_buses + self.pq_buses
        self.pq_pv_buses.sort()

        # ── Scheduled power injections (generation - load) ────────────
        self.P_sch = ((self.bus_data['P_gen_pu'] - self.bus_data['P_load_pu'])
                      .values.astype(float))
        self.Q_sch = ((self.bus_data['Q_gen_pu'] - self.bus_data['Q_load_pu'])
                      .values.astype(float))

        # ── Initial voltage state (flat start) ────────────────────────
        # All |V| = 1.0, all θ = 0  (except slack and PV with known V)
        self.V     = self.bus_data['V_mag'].values.astype(float).copy()
        self.theta = np.deg2rad(
            self.bus_data['V_angle_deg'].values.astype(float).copy()
        )

        # ── Convergence history (for plotting) ────────────────────────
        self.mismatch_history = []    # max mismatch per iteration
        self.converged        = False
        self.iterations_used  = 0

        print(f"[NRSolver] Initialized — {self.n} buses | "
              f"Slack: {[b+1 for b in self.slack_buses]} | "
              f"PV: {[b+1 for b in self.pv_buses]} | "
              f"PQ: {[b+1 for b in self.pq_buses]}")

    # ==================================================================
    # 1.  POWER INJECTION CALCULATOR
    # ==================================================================
    def _calc_power_injections(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate P_calc and Q_calc at every bus using current V and θ.

        P_i = |V_i| Σ_j |V_j|(G_ij cosθ_ij + B_ij sinθ_ij)
        Q_i = |V_i| Σ_j |V_j|(G_ij sinθ_ij - B_ij cosθ_ij)
        """
        P_calc = np.zeros(self.n)
        Q_calc = np.zeros(self.n)

        for i in range(self.n):
            for j in range(self.n):
                theta_ij = self.theta[i] - self.theta[j]
                P_calc[i] += self.V[i] * self.V[j] * (
                    self.G[i, j] * np.cos(theta_ij) +
                    self.B[i, j] * np.sin(theta_ij)
                )
                Q_calc[i] += self.V[i] * self.V[j] * (
                    self.G[i, j] * np.sin(theta_ij) -
                    self.B[i, j] * np.cos(theta_ij)
                )

        return P_calc, Q_calc

    # ==================================================================
    # 2.  MISMATCH CALCULATOR
    # ==================================================================
    def _calc_mismatch(self, P_calc: np.ndarray,
                       Q_calc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ΔP_i = P_scheduled_i - P_calculated_i   (for all non-slack buses)
        ΔQ_i = Q_scheduled_i - Q_calculated_i   (for PQ buses only)
        """
        dP = np.array([self.P_sch[i] - P_calc[i] for i in self.pq_pv_buses])
        dQ = np.array([self.Q_sch[i] - Q_calc[i] for i in self.pq_buses])
        return dP, dQ

    # ==================================================================
    # 3.  JACOBIAN MATRIX BUILDER
    # ==================================================================
    def _build_jacobian(self, P_calc: np.ndarray,
                        Q_calc: np.ndarray) -> np.ndarray:
        """
        Build the full Jacobian matrix J = [[H, N], [J_mat, L]]

        Submatrices:
          H = ∂P/∂θ       (rows: pq+pv buses, cols: pq+pv buses)
          N = ∂P/∂|V|·|V| (rows: pq+pv buses, cols: pq buses)
          J = ∂Q/∂θ       (rows: pq buses,    cols: pq+pv buses)
          L = ∂Q/∂|V|·|V| (rows: pq buses,    cols: pq buses)

        Partial derivative formulas:
        Off-diagonal (i ≠ j):
          H_ij =  |V_i||V_j|(G_ij sinθ_ij - B_ij cosθ_ij)
          N_ij =  |V_i||V_j|(G_ij cosθ_ij + B_ij sinθ_ij)
          J_ij = -|V_i||V_j|(G_ij cosθ_ij + B_ij sinθ_ij)
          L_ij =  |V_i||V_j|(G_ij sinθ_ij - B_ij cosθ_ij)

        Diagonal (i = i):
          H_ii = -Q_i - B_ii|V_i|²
          N_ii =  P_i + G_ii|V_i|²
          J_ii =  P_i - G_ii|V_i|²
          L_ii =  Q_i - B_ii|V_i|²
        """
        n_pq_pv = len(self.pq_pv_buses)
        n_pq    = len(self.pq_buses)

        # Allocate submatrices
        H     = np.zeros((n_pq_pv, n_pq_pv))
        N_mat = np.zeros((n_pq_pv, n_pq))
        J_mat = np.zeros((n_pq,    n_pq_pv))
        L     = np.zeros((n_pq,    n_pq))

        # ── Fill H and N ──────────────────────────────────────────────
        for row_idx, i in enumerate(self.pq_pv_buses):
            for col_idx, j in enumerate(self.pq_pv_buses):
                theta_ij = self.theta[i] - self.theta[j]
                if i != j:
                    H[row_idx, col_idx] = (
                        self.V[i] * self.V[j] *
                        (self.G[i, j] * np.sin(theta_ij) -
                         self.B[i, j] * np.cos(theta_ij))
                    )
                else:
                    H[row_idx, col_idx] = (
                        -Q_calc[i] - self.B[i, i] * self.V[i] ** 2
                    )

            for col_idx, j in enumerate(self.pq_buses):
                theta_ij = self.theta[i] - self.theta[j]
                if i != j:
                    N_mat[row_idx, col_idx] = (
                        self.V[i] * self.V[j] *
                        (self.G[i, j] * np.cos(theta_ij) +
                         self.B[i, j] * np.sin(theta_ij))
                    )
                else:
                    N_mat[row_idx, col_idx] = (
                        P_calc[i] + self.G[i, i] * self.V[i] ** 2
                    )

        # ── Fill J_mat and L ──────────────────────────────────────────
        for row_idx, i in enumerate(self.pq_buses):
            for col_idx, j in enumerate(self.pq_pv_buses):
                theta_ij = self.theta[i] - self.theta[j]
                if i != j:
                    J_mat[row_idx, col_idx] = (
                        -self.V[i] * self.V[j] *
                        (self.G[i, j] * np.cos(theta_ij) +
                         self.B[i, j] * np.sin(theta_ij))
                    )
                else:
                    J_mat[row_idx, col_idx] = (
                        P_calc[i] - self.G[i, i] * self.V[i] ** 2
                    )

            for col_idx, j in enumerate(self.pq_buses):
                theta_ij = self.theta[i] - self.theta[j]
                if i != j:
                    L[row_idx, col_idx] = (
                        self.V[i] * self.V[j] *
                        (self.G[i, j] * np.sin(theta_ij) -
                         self.B[i, j] * np.cos(theta_ij))
                    )
                else:
                    L[row_idx, col_idx] = (
                        Q_calc[i] - self.B[i, i] * self.V[i] ** 2
                    )

        # ── Assemble full Jacobian ────────────────────────────────────
        top    = np.hstack([H,     N_mat])
        bottom = np.hstack([J_mat, L    ])
        J_full = np.vstack([top,   bottom])

        return J_full

    # ==================================================================
    # 4.  MAIN NEWTON-RAPHSON ITERATION LOOP
    # ==================================================================
    def solve(self) -> dict:
        """
        Run Newton-Raphson iterations until convergence or max_iter.

        Returns
        -------
        dict with keys:
          converged, iterations, V, theta_deg,
          P_calc, Q_calc, mismatch_history
        """
        print(f"\n[NRSolver] Starting Newton-Raphson iterations "
              f"(tol={self.tolerance}, max_iter={self.max_iter})")
        print("-" * 60)
        print(f"  {'Iter':>4}  {'Max |ΔP| (pu)':>16}  "
              f"{'Max |ΔQ| (pu)':>16}  {'Status':>12}")
        print("-" * 60)

        for iteration in range(1, self.max_iter + 1):

            # Step 1: Calculate power injections
            P_calc, Q_calc = self._calc_power_injections()

            # Step 2: Calculate mismatches
            dP, dQ = self._calc_mismatch(P_calc, Q_calc)

            # Step 3: Check convergence
            max_dP = np.max(np.abs(dP)) if len(dP) > 0 else 0.0
            max_dQ = np.max(np.abs(dQ)) if len(dQ) > 0 else 0.0
            max_mismatch = max(max_dP, max_dQ)
            self.mismatch_history.append(max_mismatch)

            status = "Converged ✓" if max_mismatch < self.tolerance else "Iterating..."
            print(f"  {iteration:>4}  {max_dP:>16.8f}  {max_dQ:>16.8f}  {status:>12}")

            if max_mismatch < self.tolerance:
                self.converged       = True
                self.iterations_used = iteration
                print("-" * 60)
                print(f"  ✅ Converged in {iteration} iterations! "
                      f"Max mismatch = {max_mismatch:.2e} pu")
                break

            # Step 4: Build Jacobian
            J = self._build_jacobian(P_calc, Q_calc)

            # Step 5: Form mismatch vector [ΔP; ΔQ]
            mismatch_vec = np.concatenate([dP, dQ])

            # Step 6: Solve linear system J * [Δθ; Δ|V|/|V|] = [ΔP; ΔQ]
            try:
                correction = np.linalg.solve(J, mismatch_vec)
            except np.linalg.LinAlgError:
                print("  ❌ Jacobian is singular — system may be ill-conditioned!")
                break

            # Step 7: Extract corrections
            n_pq_pv = len(self.pq_pv_buses)
            d_theta = correction[:n_pq_pv]          # angle corrections (rad)
            d_V_V   = correction[n_pq_pv:]           # |ΔV|/|V| corrections

            # Step 8: Update state variables
            for idx, i in enumerate(self.pq_pv_buses):
                self.theta[i] += d_theta[idx]

            for idx, i in enumerate(self.pq_buses):
                self.V[i] += d_V_V[idx] * self.V[i]

        else:
            print(f"\n  ⚠️  Did not converge in {self.max_iter} iterations!")

        # Final power injection calculation
        P_calc, Q_calc = self._calc_power_injections()

        return {
            "converged"        : self.converged,
            "iterations"       : self.iterations_used,
            "V"                : self.V.copy(),
            "theta_rad"        : self.theta.copy(),
            "theta_deg"        : np.rad2deg(self.theta).copy(),
            "P_calc"           : P_calc,
            "Q_calc"           : Q_calc,
            "mismatch_history" : self.mismatch_history.copy(),
        }