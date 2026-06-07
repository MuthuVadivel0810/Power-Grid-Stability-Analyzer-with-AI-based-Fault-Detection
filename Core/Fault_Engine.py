"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: Fault Engine (core/fault_engine.py)

  What this does:
    - Builds sequence impedance networks (Z1, Z2, Z0)
    - Calculates fault currents for all 4 fault types using
      Symmetrical Component theory
    - Computes post-fault bus voltages across the entire network
    - Returns structured FaultResult objects for waveform generation

  ── Symmetrical Component Theory ──────────────────────────────
  Any unbalanced 3-phase system can be decomposed into:
    • Positive sequence (1) : balanced, ABC rotation
    • Negative sequence (2) : balanced, ACB rotation
    • Zero sequence     (0) : all three phases in phase

  ── Sequence Impedances ───────────────────────────────────────
  Thevenin impedance seen at fault bus k in each sequence:
    Z1_th = Z_bus1[k,k]   where Z_bus1 = inv(Ybus_positive)
    Z2_th ≈ Z1_th          (for transmission lines Z2 = Z1)
    Z0_th = 3 × Z1_th      (higher due to ground return path)

  ── Fault Current Formulas ────────────────────────────────────
  Pre-fault voltage at fault bus: Vf ≈ V[k] from load flow

  3-Phase (3Φ) — Symmetrical:
    Ia1  = Vf / Z1
    |If| = |Vf / Z1|

  Line-to-Ground (L-G) on Phase A:
    Ia1  = Vf / (Z1 + Z2 + Z0)
    If   = 3 × Ia1

  Line-to-Line (L-L) between Phases B and C:
    Ia1  = Vf / (Z1 + Z2)
    |If| = √3 × |Ia1|

  Double Line-to-Ground (L-L-G) on Phases B and C:
    Ia1  = Vf / (Z1 + Z2‖Z0)
    where Z2‖Z0 = Z2×Z0 / (Z2+Z0)
=============================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List


# ── Fault type constants ──────────────────────────────────────────────────────
FAULT_TYPES = {
    0: "No_Fault",
    1: "Three_Phase",
    2: "Line_to_Ground",
    3: "Line_to_Line",
    4: "Double_Line_to_Ground",
}

FAULT_LABELS = {v: k for k, v in FAULT_TYPES.items()}


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class FaultResult:
    """
    Stores all computed quantities for a single fault event.

    Fields
    ------
    fault_type     : str   – fault type name
    fault_bus      : int   – 1-based bus number where fault occurs
    fault_resistance: float – fault resistance Rf in pu (0 = bolted fault)
    Vf             : complex – pre-fault voltage at fault bus
    Z1, Z2, Z0     : complex – Thevenin sequence impedances at fault bus
    Ia0, Ia1, Ia2  : complex – sequence fault currents
    Ia, Ib, Ic     : complex – phase fault currents (from sequence currents)
    If_mag         : float  – total fault current magnitude (pu)
    V_bus_prefault : np.ndarray – pre-fault bus voltages (complex)
    V_bus_postfault: np.ndarray – post-fault bus voltages (complex)
    voltage_sag    : np.ndarray – per-bus voltage sag depth (pu)
    """
    fault_type     : str
    fault_bus      : int
    fault_resistance: float
    Vf             : complex
    Z1             : complex
    Z2             : complex
    Z0             : complex
    Ia0            : complex = 0j
    Ia1            : complex = 0j
    Ia2            : complex = 0j
    Ia             : complex = 0j
    Ib             : complex = 0j
    Ic             : complex = 0j
    If_mag         : float   = 0.0
    V_bus_prefault : np.ndarray = field(default_factory=lambda: np.array([]))
    V_bus_postfault: np.ndarray = field(default_factory=lambda: np.array([]))
    voltage_sag    : np.ndarray = field(default_factory=lambda: np.array([]))


class FaultEngine:
    """
    Computes fault currents and post-fault voltages using
    Symmetrical Component theory.

    Parameters
    ----------
    Ybus      : np.ndarray – positive-sequence Ybus from YbusBuilder
    V_prefault: np.ndarray – complex pre-fault bus voltages from NR solver
    Z0_factor : float      – Z0 = Z0_factor × Z1  (default 3.0)
    """

    def __init__(self, Ybus: np.ndarray, V_prefault: np.ndarray,
                 Z0_factor: float = 3.0):
        self.Ybus       = Ybus
        self.n          = Ybus.shape[0]
        self.Z0_factor  = Z0_factor

        # Store pre-fault voltages as complex phasors
        self.V_pre = V_prefault.astype(complex)

        # ── Build bus impedance matrix Zbus = inv(Ybus) ───────────────
        # Zbus[k,k] = Thevenin impedance seen looking into bus k
        self.Zbus1 = np.linalg.inv(Ybus)           # positive-sequence Zbus
        self.Zbus2 = self.Zbus1.copy()             # Z2 = Z1 for lines
        self.Zbus0 = self.Zbus1 * Z0_factor        # Z0 = Z0_factor × Z1

        print(f"[FaultEngine] Initialized — {self.n} buses, "
              f"Z0 = {Z0_factor}×Z1")
        print(f"  Zbus diagonal (|Z1_th| pu):")
        for i in range(self.n):
            print(f"    Bus-{i+1}: |Z1| = {abs(self.Zbus1[i,i]):.4f} pu")

    # ==================================================================
    # CORE FAULT CALCULATORS
    # ==================================================================

    def _get_sequence_Z(self, bus_idx: int, Rf: float = 0.0):
        """
        Return Thevenin sequence impedances at bus_idx, including
        optional fault resistance Rf (bolted fault: Rf=0).
        """
        Z1 = self.Zbus1[bus_idx, bus_idx] + Rf
        Z2 = self.Zbus2[bus_idx, bus_idx]
        Z0 = self.Zbus0[bus_idx, bus_idx] + 3 * Rf   # 3Rf in zero sequence
        return Z1, Z2, Z0

    # ------------------------------------------------------------------
    def calc_three_phase_fault(self, bus_idx: int,
                               Rf: float = 0.0) -> FaultResult:
        """
        3-Phase (3Φ) Symmetrical Fault.

        Only positive sequence network is involved:
          Ia1 = Vf / (Z1 + Rf)
          Ia  = Ia1,  Ib = a²·Ia1,  Ic = a·Ia1
          where a = e^(j2π/3)
        """
        Vf = self.V_pre[bus_idx]
        Z1, Z2, Z0 = self._get_sequence_Z(bus_idx, Rf)

        Ia1 = Vf / Z1
        Ia0 = 0 + 0j
        Ia2 = 0 + 0j

        a = np.exp(1j * 2 * np.pi / 3)      # 120° operator
        Ia = Ia0 + Ia1 + Ia2
        Ib = Ia0 + (a**2) * Ia1 + a * Ia2
        Ic = Ia0 + a * Ia1 + (a**2) * Ia2

        result = FaultResult(
            fault_type="Three_Phase", fault_bus=bus_idx + 1,
            fault_resistance=Rf, Vf=Vf, Z1=Z1, Z2=Z2, Z0=Z0,
            Ia0=Ia0, Ia1=Ia1, Ia2=Ia2,
            Ia=Ia, Ib=Ib, Ic=Ic, If_mag=abs(Ia1)
        )
        self._calc_post_fault_voltages(result, bus_idx)
        return result

    # ------------------------------------------------------------------
    def calc_lg_fault(self, bus_idx: int,
                      Rf: float = 0.0) -> FaultResult:
        """
        Line-to-Ground (L-G) Fault on Phase A.

        All three sequence networks connected in SERIES:
          Ia1 = Vf / (Z1 + Z2 + Z0)    ← series connection
          Ia0 = Ia2 = Ia1
          Ia  = 3·Ia1  (ground fault current)
          Ib  = Ic = 0
        """
        Vf = self.V_pre[bus_idx]
        Z1, Z2, Z0 = self._get_sequence_Z(bus_idx, Rf)

        Ia1 = Vf / (Z1 + Z2 + Z0)
        Ia0 = Ia1
        Ia2 = Ia1

        a   = np.exp(1j * 2 * np.pi / 3)
        Ia  = Ia0 + Ia1 + Ia2            # = 3·Ia1
        Ib  = Ia0 + (a**2) * Ia1 + a * Ia2
        Ic  = Ia0 + a * Ia1 + (a**2) * Ia2

        result = FaultResult(
            fault_type="Line_to_Ground", fault_bus=bus_idx + 1,
            fault_resistance=Rf, Vf=Vf, Z1=Z1, Z2=Z2, Z0=Z0,
            Ia0=Ia0, Ia1=Ia1, Ia2=Ia2,
            Ia=Ia, Ib=Ib, Ic=Ic, If_mag=abs(Ia)
        )
        self._calc_post_fault_voltages(result, bus_idx)
        return result

    # ------------------------------------------------------------------
    def calc_ll_fault(self, bus_idx: int,
                      Rf: float = 0.0) -> FaultResult:
        """
        Line-to-Line (L-L) Fault between Phases B and C.

        Positive and negative sequence connected in PARALLEL (no zero seq):
          Ia1  =  Vf / (Z1 + Z2)
          Ia2  = -Ia1
          Ia0  =  0
          Ia   =  0  (phase A unaffected)
          Ib   = -Ic = -j√3·Ia1
        """
        Vf = self.V_pre[bus_idx]
        Z1, Z2, Z0 = self._get_sequence_Z(bus_idx, Rf)

        Ia1 = Vf / (Z1 + Z2)
        Ia2 = -Ia1
        Ia0 = 0 + 0j

        a   = np.exp(1j * 2 * np.pi / 3)
        Ia  = Ia0 + Ia1 + Ia2                       # ≈ 0
        Ib  = Ia0 + (a**2) * Ia1 + a * Ia2
        Ic  = Ia0 + a * Ia1 + (a**2) * Ia2

        result = FaultResult(
            fault_type="Line_to_Line", fault_bus=bus_idx + 1,
            fault_resistance=Rf, Vf=Vf, Z1=Z1, Z2=Z2, Z0=Z0,
            Ia0=Ia0, Ia1=Ia1, Ia2=Ia2,
            Ia=Ia, Ib=Ib, Ic=Ic, If_mag=abs(Ib)
        )
        self._calc_post_fault_voltages(result, bus_idx)
        return result

    # ------------------------------------------------------------------
    def calc_llg_fault(self, bus_idx: int,
                       Rf: float = 0.0) -> FaultResult:
        """
        Double Line-to-Ground (L-L-G) Fault on Phases B and C.

        Positive sequence connects to PARALLEL combination of Z2 and Z0:
          Z_parallel = Z2 × Z0 / (Z2 + Z0)
          Ia1  = Vf / (Z1 + Z_parallel)
          Ia2  = -Ia1 × Z0 / (Z2 + Z0)
          Ia0  = -Ia1 × Z2 / (Z2 + Z0)
        """
        Vf = self.V_pre[bus_idx]
        Z1, Z2, Z0 = self._get_sequence_Z(bus_idx, Rf)

        Z20 = (Z2 * Z0) / (Z2 + Z0)               # parallel combination
        Ia1 = Vf / (Z1 + Z20)
        Ia2 = -Ia1 * Z0 / (Z2 + Z0)
        Ia0 = -Ia1 * Z2 / (Z2 + Z0)

        a   = np.exp(1j * 2 * np.pi / 3)
        Ia  = Ia0 + Ia1 + Ia2
        Ib  = Ia0 + (a**2) * Ia1 + a * Ia2
        Ic  = Ia0 + a * Ia1 + (a**2) * Ia2

        result = FaultResult(
            fault_type="Double_Line_to_Ground", fault_bus=bus_idx + 1,
            fault_resistance=Rf, Vf=Vf, Z1=Z1, Z2=Z2, Z0=Z0,
            Ia0=Ia0, Ia1=Ia1, Ia2=Ia2,
            Ia=Ia, Ib=Ib, Ic=Ic, If_mag=abs(Ib) + abs(Ic)
        )
        self._calc_post_fault_voltages(result, bus_idx)
        return result

    # ==================================================================
    # POST-FAULT BUS VOLTAGE CALCULATOR
    # ==================================================================
    def _calc_post_fault_voltages(self, result: FaultResult, fault_bus_idx: int):
        """
        Calculate voltage at every bus after the fault using superposition.

        Post-fault voltage at bus j (positive sequence):
          V1_j = V_pre_j - Zbus1[j, fault_bus] × Ia1

        Similarly for V2 and V0.
        Then convert back to phase voltages using:
          [Va]   [1  1  1 ] [V0]
          [Vb] = [1  a²  a] [V1]
          [Vc]   [1  a  a²] [V2]
        """
        k = fault_bus_idx
        a = np.exp(1j * 2 * np.pi / 3)

        # Sequence voltage at every bus j
        V0_post = np.array([-self.Zbus0[j, k] * result.Ia0
                            for j in range(self.n)])
        V1_post = np.array([self.V_pre[j] - self.Zbus1[j, k] * result.Ia1
                            for j in range(self.n)])
        V2_post = np.array([-self.Zbus2[j, k] * result.Ia2
                            for j in range(self.n)])

        # Convert to phase voltages
        Va_post = V0_post + V1_post + V2_post
        Vb_post = V0_post + (a**2) * V1_post + a * V2_post
        Vc_post = V0_post + a * V1_post + (a**2) * V2_post

        # Average phase voltage magnitude at each bus
        V_post_mag = (np.abs(Va_post) + np.abs(Vb_post) + np.abs(Vc_post)) / 3.0

        result.V_bus_prefault  = np.abs(self.V_pre)
        result.V_bus_postfault = V_post_mag
        result.voltage_sag     = np.abs(self.V_pre) - V_post_mag

    # ==================================================================
    # DISPATCHER: run any fault type by name
    # ==================================================================
    def run_fault(self, fault_type: str, bus_idx: int,
                  Rf: float = 0.0) -> FaultResult:
        """
        Dispatch to the correct fault calculator.

        Parameters
        ----------
        fault_type : one of 'Three_Phase', 'Line_to_Ground',
                     'Line_to_Line', 'Double_Line_to_Ground'
        bus_idx    : 0-based bus index
        Rf         : fault resistance in pu
        """
        calculators = {
            "Three_Phase"          : self.calc_three_phase_fault,
            "Line_to_Ground"       : self.calc_lg_fault,
            "Line_to_Line"         : self.calc_ll_fault,
            "Double_Line_to_Ground": self.calc_llg_fault,
        }
        if fault_type not in calculators:
            raise ValueError(f"Unknown fault type: {fault_type}. "
                             f"Choose from {list(calculators.keys())}")
        return calculators[fault_type](bus_idx, Rf)

    # ==================================================================
    # PRINT FAULT REPORT
    # ==================================================================
    def print_fault_report(self, result: FaultResult):
        print(f"\n{'='*60}")
        print(f"  FAULT ANALYSIS REPORT")
        print(f"{'='*60}")
        print(f"  Fault Type       : {result.fault_type}")
        print(f"  Fault Bus        : Bus-{result.fault_bus}")
        print(f"  Fault Resistance : {result.fault_resistance:.4f} pu")
        print(f"  Pre-fault Voltage: {abs(result.Vf):.4f} pu ∠"
              f"{np.rad2deg(np.angle(result.Vf)):.2f}°")
        print(f"\n  Sequence Impedances (Thevenin at fault bus):")
        print(f"    Z1 = {result.Z1.real:.4f} + j{result.Z1.imag:.4f} pu")
        print(f"    Z2 = {result.Z2.real:.4f} + j{result.Z2.imag:.4f} pu")
        print(f"    Z0 = {result.Z0.real:.4f} + j{result.Z0.imag:.4f} pu")
        print(f"\n  Sequence Fault Currents:")
        print(f"    Ia1 = {abs(result.Ia1):.4f} pu ∠{np.rad2deg(np.angle(result.Ia1)):.2f}°")
        print(f"    Ia2 = {abs(result.Ia2):.4f} pu ∠{np.rad2deg(np.angle(result.Ia2)):.2f}°")
        print(f"    Ia0 = {abs(result.Ia0):.4f} pu ∠{np.rad2deg(np.angle(result.Ia0)):.2f}°")
        print(f"\n  Phase Fault Currents:")
        print(f"    Ia = {abs(result.Ia):.4f} pu  "
              f"Ib = {abs(result.Ib):.4f} pu  "
              f"Ic = {abs(result.Ic):.4f} pu")
        print(f"    Total |If| = {result.If_mag:.4f} pu")
        print(f"\n  Post-Fault Bus Voltages:")
        for i, (vpre, vpost, vsag) in enumerate(zip(
                result.V_bus_prefault, result.V_bus_postfault, result.voltage_sag)):
            flag = "  ⚠ SAG" if vsag > 0.05 else ""
            print(f"    Bus-{i+1}: {vpre:.4f} pu → {vpost:.4f} pu  "
                  f"(sag = {vsag:.4f} pu){flag}")
        print(f"{'='*60}")