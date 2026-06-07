"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: Waveform Generator (core/waveform_generator.py)

  What this does:
    - Synthesizes realistic 3-phase voltage & current waveforms
    - Injects fault signatures based on FaultResult data
    - Each waveform has 3 regions:
        PRE-FAULT  : balanced 3-phase sinusoids at 50 Hz
        DURING FAULT: voltage sag + current spike + harmonics
                     + phase imbalance (unique per fault type)
        POST-FAULT : gradual voltage recovery

  Fault Signatures injected per fault type:
    Three_Phase          : symmetric voltage sag, high symmetric current
    Line_to_Ground       : phase A sag + large 3rd harmonic (zero seq)
    Line_to_Line         : phase B/C sag + 2nd harmonic (negative seq)
    Double_Line_to_Ground: phase B/C sag + mixed 2nd & 3rd harmonics
    No_Fault             : clean balanced waveform (small noise only)
=============================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from Core.Fault_Engine import FaultResult


@dataclass
class Waveform:
    """
    Container for a 3-phase voltage + current waveform.

    Fields
    ------
    t         : time array (seconds)
    Va, Vb, Vc: phase voltages (pu)
    Ia, Ib, Ic: phase currents (pu)
    fault_type: label string
    fault_bus : bus number
    label     : integer class label (0..4)
    """
    t          : np.ndarray
    Va         : np.ndarray
    Vb         : np.ndarray
    Vc         : np.ndarray
    Ia         : np.ndarray
    Ib         : np.ndarray
    Ic         : np.ndarray
    fault_type : str
    fault_bus  : int
    label      : int


class WaveformGenerator:
    """
    Generates 3-phase time-domain voltage & current waveforms
    with embedded fault signatures.

    Parameters
    ----------
    fs          : sampling frequency in Hz (default 6400 — 128 samples/cycle)
    duration    : total waveform duration in seconds (default 0.1 s = 5 cycles)
    fault_start : fault onset time in seconds (default 0.04 s = 2 cycles in)
    fault_dur   : fault duration in seconds (default 0.04 s = 2 cycles)
    freq        : system frequency in Hz (default 50 Hz)
    noise_std   : standard deviation of background noise (default 0.005 pu)
    """

    LABEL_MAP = {
        "No_Fault"             : 0,
        "Three_Phase"          : 1,
        "Line_to_Ground"       : 2,
        "Line_to_Line"         : 3,
        "Double_Line_to_Ground": 4,
    }

    def __init__(self, fs: float = 6400.0, duration: float = 0.1,
                 fault_start: float = 0.04, fault_dur: float = 0.04,
                 freq: float = 50.0, noise_std: float = 0.005):
        self.fs          = fs
        self.duration    = duration
        self.fault_start = fault_start
        self.fault_end   = fault_start + fault_dur
        self.freq        = freq
        self.omega       = 2 * np.pi * freq
        self.noise_std   = noise_std

        # Time array
        self.t = np.arange(0, duration, 1.0 / fs)
        self.N = len(self.t)

        # Fault window masks
        self._pre   = self.t < fault_start
        self._fault = (self.t >= fault_start) & (self.t < self.fault_end)
        self._post  = self.t >= self.fault_end

    # ==================================================================
    # BASE WAVEFORMS
    # ==================================================================
    def _balanced_voltage(self, V_mag: float = 1.0):
        """Clean balanced 3-phase voltage waveforms."""
        Va = V_mag * np.sqrt(2) * np.sin(self.omega * self.t)
        Vb = V_mag * np.sqrt(2) * np.sin(self.omega * self.t - 2*np.pi/3)
        Vc = V_mag * np.sqrt(2) * np.sin(self.omega * self.t + 2*np.pi/3)
        return Va, Vb, Vc

    def _balanced_current(self, I_mag: float = 0.3, phi: float = 0.3):
        """Balanced load current (lagging power factor)."""
        Ia = I_mag * np.sqrt(2) * np.sin(self.omega * self.t - phi)
        Ib = I_mag * np.sqrt(2) * np.sin(self.omega * self.t - phi - 2*np.pi/3)
        Ic = I_mag * np.sqrt(2) * np.sin(self.omega * self.t - phi + 2*np.pi/3)
        return Ia, Ib, Ic

    def _add_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add Gaussian white noise to signal."""
        return signal + np.random.normal(0, self.noise_std, len(signal))

    def _recovery_envelope(self) -> np.ndarray:
        """
        Post-fault voltage recovery: exponential rise back to nominal.
        Shape: 0 → 1 over post-fault duration.
        """
        env = np.ones(self.N)
        if np.any(self._post):
            t_post = self.t[self._post] - self.fault_end
            tau    = 0.02    # recovery time constant (seconds)
            env[self._post] = 1 - np.exp(-t_post / tau)
        return env

    # ==================================================================
    # NO FAULT WAVEFORM
    # ==================================================================
    def generate_no_fault(self, bus_idx: int = 0,
                          V_mag: float = 1.0) -> Waveform:
        """Clean balanced waveform — class label 0."""
        Va, Vb, Vc = self._balanced_voltage(V_mag)
        Ia, Ib, Ic = self._balanced_current()
        return Waveform(
            t=self.t, Va=self._add_noise(Va), Vb=self._add_noise(Vb),
            Vc=self._add_noise(Vc), Ia=self._add_noise(Ia),
            Ib=self._add_noise(Ib), Ic=self._add_noise(Ic),
            fault_type="No_Fault", fault_bus=bus_idx + 1, label=0
        )

    # ==================================================================
    # FAULT WAVEFORM GENERATORS
    # ==================================================================

    def _apply_fault_to_waveforms(self, Va, Vb, Vc, Ia, Ib, Ic,
                                   result: FaultResult) -> tuple:
        """
        Modify pre-fault balanced waveforms by injecting fault signatures
        specific to the fault type returned by FaultEngine.
        """
        ft  = result.fault_type
        k   = result.fault_bus - 1    # 0-based fault bus index
        sag = result.voltage_sag      # per-bus sag array

        # ── Compute sag depths for the fault bus ──────────────────────
        Va_sag = float(np.clip(sag[k] * 1.5, 0, 0.95))
        Vb_sag = float(np.clip(sag[k] * 1.2, 0, 0.95))
        Vc_sag = float(np.clip(sag[k] * 1.2, 0, 0.95))
        If_pu  = float(np.clip(result.If_mag, 0.5, 15.0))

        rec = self._recovery_envelope()

        # ── THREE-PHASE FAULT ─────────────────────────────────────────
        if ft == "Three_Phase":
            # All 3 phases sag symmetrically, large balanced current
            Va[self._fault] *= (1 - Va_sag)
            Vb[self._fault] *= (1 - Va_sag)
            Vc[self._fault] *= (1 - Va_sag)
            # High symmetric fault current
            Ia[self._fault] += If_pu * np.sqrt(2) * np.sin(
                self.omega * self.t[self._fault])
            Ib[self._fault] += If_pu * np.sqrt(2) * np.sin(
                self.omega * self.t[self._fault] - 2*np.pi/3)
            Ic[self._fault] += If_pu * np.sqrt(2) * np.sin(
                self.omega * self.t[self._fault] + 2*np.pi/3)

        # ── LINE-TO-GROUND FAULT ──────────────────────────────────────
        elif ft == "Line_to_Ground":
            # Phase A deep sag, B & C slightly affected
            Va[self._fault] *= (1 - Va_sag * 1.2)
            Vb[self._fault] *= (1 - Vb_sag * 0.15)
            Vc[self._fault] *= (1 - Vc_sag * 0.15)
            # Large ground current in Ia + 3rd harmonic (zero-sequence signature)
            Ia[self._fault] += If_pu * np.sqrt(2) * np.sin(
                self.omega * self.t[self._fault])
            Ia[self._fault] += 0.3 * If_pu * np.sin(     # 3rd harmonic
                3 * self.omega * self.t[self._fault])
            Vb[self._fault] += 0.08 * np.sin(            # zero-seq voltage rise on B
                3 * self.omega * self.t[self._fault])
            Vc[self._fault] += 0.08 * np.sin(
                3 * self.omega * self.t[self._fault])

        # ── LINE-TO-LINE FAULT ────────────────────────────────────────
        elif ft == "Line_to_Line":
            # Phase B and C sag + phase shift, A unaffected
            Va[self._fault] *= (1 - Va_sag * 0.05)
            Vb[self._fault] *= (1 - Vb_sag * 0.9)
            Vc[self._fault] *= (1 - Vc_sag * 0.9)
            # High currents in B and C (equal & opposite), 2nd harmonic (neg-seq)
            I_ll = If_pu * np.sqrt(2) * 0.866
            Ib[self._fault] += I_ll * np.sin(
                self.omega * self.t[self._fault] - 2*np.pi/3)
            Ic[self._fault] -= I_ll * np.sin(
                self.omega * self.t[self._fault] - 2*np.pi/3)
            Ib[self._fault] += 0.2 * If_pu * np.sin(     # 2nd harmonic
                2 * self.omega * self.t[self._fault])
            Ic[self._fault] -= 0.2 * If_pu * np.sin(
                2 * self.omega * self.t[self._fault])

        # ── DOUBLE LINE-TO-GROUND FAULT ───────────────────────────────
        elif ft == "Double_Line_to_Ground":
            # B and C sag severely, A less affected
            Va[self._fault] *= (1 - Va_sag * 0.3)
            Vb[self._fault] *= (1 - Vb_sag * 1.0)
            Vc[self._fault] *= (1 - Vc_sag * 1.0)
            # Mixed harmonics: 2nd (neg-seq) + 3rd (zero-seq)
            Ib[self._fault] += If_pu * 0.7 * np.sqrt(2) * np.sin(
                self.omega * self.t[self._fault])
            Ic[self._fault] += If_pu * 0.7 * np.sqrt(2) * np.sin(
                self.omega * self.t[self._fault])
            Ib[self._fault] += 0.25 * If_pu * np.sin(    # 2nd harmonic
                2 * self.omega * self.t[self._fault])
            Ib[self._fault] += 0.20 * If_pu * np.sin(    # 3rd harmonic
                3 * self.omega * self.t[self._fault])
            Ic[self._fault] += 0.25 * If_pu * np.sin(
                2 * self.omega * self.t[self._fault])
            Ic[self._fault] += 0.20 * If_pu * np.sin(
                3 * self.omega * self.t[self._fault])

        # ── POST-FAULT RECOVERY ───────────────────────────────────────
        # Voltage recovers gradually after fault clears
        Va[self._post] *= (0.85 + 0.15 * rec[self._post])
        Vb[self._post] *= (0.85 + 0.15 * rec[self._post])
        Vc[self._post] *= (0.85 + 0.15 * rec[self._post])

        return Va, Vb, Vc, Ia, Ib, Ic

    # ------------------------------------------------------------------
    def generate_fault_waveform(self, result: FaultResult) -> Waveform:
        """
        Main method: generate full 3-phase waveform for a given FaultResult.

        Returns a Waveform object with Va, Vb, Vc, Ia, Ib, Ic arrays.
        """
        V_pre_mag = float(np.mean(result.V_bus_prefault))
        Va, Vb, Vc = self._balanced_voltage(V_pre_mag)
        Ia, Ib, Ic = self._balanced_current()

        Va, Vb, Vc, Ia, Ib, Ic = self._apply_fault_to_waveforms(
            Va, Vb, Vc, Ia, Ib, Ic, result
        )

        return Waveform(
            t=self.t.copy(),
            Va=self._add_noise(Va), Vb=self._add_noise(Vb),
            Vc=self._add_noise(Vc), Ia=self._add_noise(Ia),
            Ib=self._add_noise(Ib), Ic=self._add_noise(Ic),
            fault_type=result.fault_type,
            fault_bus=result.fault_bus,
            label=self.LABEL_MAP[result.fault_type]
        )