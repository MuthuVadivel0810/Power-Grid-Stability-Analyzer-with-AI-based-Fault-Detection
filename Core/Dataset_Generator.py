"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: Dataset Generator (core/dataset_generator.py)

  What this does:
    - Runs all 4 fault types × all 5 buses × multiple Rf values
    - Generates time-domain waveforms for each combination
    - Extracts engineered features from each waveform:
        • RMS voltage (Va, Vb, Vc)
        • RMS current (Ia, Ib, Ic)
        • Voltage sag depths per phase
        • Peak current per phase
        • FFT — harmonic magnitudes (2nd, 3rd, 5th)
        • Total Harmonic Distortion (THD)
        • Negative sequence component |V2| / |V1|
        • Zero sequence component    |V0| / |V1|
        • Current imbalance factor
        • Voltage imbalance factor
    - Saves labeled dataset as CSV for the ML phase

  Dataset stats:
    4 fault types × 5 buses × 40 Rf values = 800 fault samples
    + No-fault: 5 buses × 40 variations   = 200 normal samples
    Total = 1000 labeled samples
=============================================================
"""

import numpy as np
import pandas as pd
from typing import List
from Core.Waveform_Generator import Waveform


class FeatureExtractor:
    """
    Extracts numerical features from a 3-phase waveform for ML training.

    All features are physics-motivated:
      - RMS values capture energy content
      - Harmonics reveal distortion signatures unique to fault type
      - Sequence components directly map to fault type (theory-based)
      - Imbalance indices quantify asymmetry
    """

    def __init__(self, fs: float = 6400.0, freq: float = 50.0):
        self.fs   = fs
        self.freq = freq
        self.T    = 1.0 / freq      # one period (seconds)

    # ------------------------------------------------------------------
    def _rms(self, signal: np.ndarray) -> float:
        """Root Mean Square of a signal."""
        return float(np.sqrt(np.mean(signal ** 2)))

    def _rms_window(self, signal: np.ndarray, t: np.ndarray,
                    t_start: float, t_end: float) -> float:
        """RMS over a specific time window."""
        mask = (t >= t_start) & (t < t_end)
        seg  = signal[mask]
        return self._rms(seg) if len(seg) > 0 else 0.0

    def _harmonic_magnitude(self, signal: np.ndarray,
                             harmonic: int) -> float:
        """
        Magnitude of the n-th harmonic using FFT.
        Returns value normalized to fundamental magnitude.
        """
        N   = len(signal)
        fft = np.fft.rfft(signal) / N
        mag = np.abs(fft)
        freqs = np.fft.rfftfreq(N, d=1.0/self.fs)

        # Find bin closest to fundamental and harmonic frequency
        f_fund    = self.freq
        f_harm    = harmonic * self.freq
        fund_idx  = np.argmin(np.abs(freqs - f_fund))
        harm_idx  = np.argmin(np.abs(freqs - f_harm))

        fund_mag  = mag[fund_idx] + 1e-9   # avoid division by zero
        return float(mag[harm_idx] / fund_mag)

    def _thd(self, signal: np.ndarray, n_harmonics: int = 10) -> float:
        """
        Total Harmonic Distortion (THD) of a signal.
        THD = sqrt(sum(V_n²)) / V1  for n = 2, 3, ..., n_harmonics
        """
        N     = len(signal)
        fft   = np.fft.rfft(signal) / N
        mag   = np.abs(fft)
        freqs = np.fft.rfftfreq(N, d=1.0/self.fs)

        fund_idx  = np.argmin(np.abs(freqs - self.freq))
        fund_mag  = mag[fund_idx] + 1e-9

        harm_sum = 0.0
        for h in range(2, n_harmonics + 1):
            f_h    = h * self.freq
            h_idx  = np.argmin(np.abs(freqs - f_h))
            harm_sum += mag[h_idx] ** 2

        return float(np.sqrt(harm_sum) / fund_mag)

    def _symmetrical_components(self, Va: float, Vb: float,
                                 Vc: float) -> tuple:
        """
        Compute symmetrical component magnitudes from 3 phase RMS values.
        Uses approximate method based on magnitudes only.

        Returns (|V0|/|V1|, |V2|/|V1|)
        """
        a = np.exp(1j * 2 * np.pi / 3)
        # Treat RMS values as phasor magnitudes with 120° separation
        Va_ph = complex(Va, 0)
        Vb_ph = Va_ph * (a ** 2)
        Vc_ph = Va_ph * a

        V0 = (Va_ph + Vb_ph + Vc_ph) / 3.0
        V1 = (Va_ph + a * Vb_ph + (a**2) * Vc_ph) / 3.0
        V2 = (Va_ph + (a**2) * Vb_ph + a * Vc_ph) / 3.0

        V1_mag = abs(V1) + 1e-9
        return float(abs(V0) / V1_mag), float(abs(V2) / V1_mag)

    # ------------------------------------------------------------------
    def extract(self, wf: Waveform) -> dict:
        """
        Extract all features from a Waveform object.

        Returns a dict of {feature_name: float_value}.
        """
        t           = wf.t
        fault_start = 0.04
        fault_end   = 0.08

        # ── Window-based RMS ──────────────────────────────────────────
        Va_rms_pre  = self._rms_window(wf.Va, t, 0.0,         fault_start)
        Vb_rms_pre  = self._rms_window(wf.Vb, t, 0.0,         fault_start)
        Vc_rms_pre  = self._rms_window(wf.Vc, t, 0.0,         fault_start)
        Va_rms_flt  = self._rms_window(wf.Va, t, fault_start, fault_end)
        Vb_rms_flt  = self._rms_window(wf.Vb, t, fault_start, fault_end)
        Vc_rms_flt  = self._rms_window(wf.Vc, t, fault_start, fault_end)
        Ia_rms_flt  = self._rms_window(wf.Ia, t, fault_start, fault_end)
        Ib_rms_flt  = self._rms_window(wf.Ib, t, fault_start, fault_end)
        Ic_rms_flt  = self._rms_window(wf.Ic, t, fault_start, fault_end)

        # ── Voltage sag depths ────────────────────────────────────────
        Va_sag = float(np.clip((Va_rms_pre - Va_rms_flt) / (Va_rms_pre + 1e-9), 0, 1))
        Vb_sag = float(np.clip((Vb_rms_pre - Vb_rms_flt) / (Vb_rms_pre + 1e-9), 0, 1))
        Vc_sag = float(np.clip((Vc_rms_pre - Vc_rms_flt) / (Vc_rms_pre + 1e-9), 0, 1))

        # ── Peak current during fault ─────────────────────────────────
        mask     = (t >= fault_start) & (t < fault_end)
        Ia_peak  = float(np.max(np.abs(wf.Ia[mask]))) if np.any(mask) else 0.0
        Ib_peak  = float(np.max(np.abs(wf.Ib[mask]))) if np.any(mask) else 0.0
        Ic_peak  = float(np.max(np.abs(wf.Ic[mask]))) if np.any(mask) else 0.0
        I_max    = max(Ia_peak, Ib_peak, Ic_peak)

        # ── Harmonic analysis (on fault window) ───────────────────────
        Va_fault_seg = wf.Va[mask] if np.any(mask) else wf.Va
        Ia_fault_seg = wf.Ia[mask] if np.any(mask) else wf.Ia
        Ib_fault_seg = wf.Ib[mask] if np.any(mask) else wf.Ib

        Va_H2 = self._harmonic_magnitude(Va_fault_seg, 2)   # negative-seq indicator
        Va_H3 = self._harmonic_magnitude(Va_fault_seg, 3)   # zero-seq indicator
        Va_H5 = self._harmonic_magnitude(Va_fault_seg, 5)
        Ia_H2 = self._harmonic_magnitude(Ia_fault_seg, 2)
        Ia_H3 = self._harmonic_magnitude(Ia_fault_seg, 3)
        Ib_H2 = self._harmonic_magnitude(Ib_fault_seg, 2)
        Ib_H3 = self._harmonic_magnitude(Ib_fault_seg, 3)

        # ── THD ───────────────────────────────────────────────────────
        Va_thd = self._thd(Va_fault_seg)
        Ia_thd = self._thd(Ia_fault_seg)

        # ── Symmetrical components ────────────────────────────────────
        V0_ratio, V2_ratio = self._symmetrical_components(
            Va_rms_flt, Vb_rms_flt, Vc_rms_flt)

        # ── Imbalance factors ─────────────────────────────────────────
        V_mean = (Va_rms_flt + Vb_rms_flt + Vc_rms_flt) / 3.0 + 1e-9
        V_imbalance = float(max(abs(Va_rms_flt - V_mean),
                                abs(Vb_rms_flt - V_mean),
                                abs(Vc_rms_flt - V_mean)) / V_mean)

        I_mean = (Ia_rms_flt + Ib_rms_flt + Ic_rms_flt) / 3.0 + 1e-9
        I_imbalance = float(max(abs(Ia_rms_flt - I_mean),
                                abs(Ib_rms_flt - I_mean),
                                abs(Ic_rms_flt - I_mean)) / I_mean)

        # ── Phase current ratio (B/A and C/A) ─────────────────────────
        Ib_Ia_ratio = float(Ib_rms_flt / (Ia_rms_flt + 1e-9))
        Ic_Ia_ratio = float(Ic_rms_flt / (Ia_rms_flt + 1e-9))

        return {
            # Voltage features
            "Va_rms_pre"    : round(Va_rms_pre, 5),
            "Va_rms_flt"    : round(Va_rms_flt, 5),
            "Vb_rms_flt"    : round(Vb_rms_flt, 5),
            "Vc_rms_flt"    : round(Vc_rms_flt, 5),
            "Va_sag"        : round(Va_sag, 5),
            "Vb_sag"        : round(Vb_sag, 5),
            "Vc_sag"        : round(Vc_sag, 5),
            # Current features
            "Ia_rms_flt"    : round(Ia_rms_flt, 5),
            "Ib_rms_flt"    : round(Ib_rms_flt, 5),
            "Ic_rms_flt"    : round(Ic_rms_flt, 5),
            "Ia_peak"       : round(Ia_peak, 5),
            "Ib_peak"       : round(Ib_peak, 5),
            "Ic_peak"       : round(Ic_peak, 5),
            "I_max_peak"    : round(I_max,   5),
            # Harmonic features
            "Va_H2"         : round(Va_H2, 5),
            "Va_H3"         : round(Va_H3, 5),
            "Va_H5"         : round(Va_H5, 5),
            "Ia_H2"         : round(Ia_H2, 5),
            "Ia_H3"         : round(Ia_H3, 5),
            "Ib_H2"         : round(Ib_H2, 5),
            "Ib_H3"         : round(Ib_H3, 5),
            # THD
            "Va_thd"        : round(Va_thd, 5),
            "Ia_thd"        : round(Ia_thd, 5),
            # Sequence components
            "V0_ratio"      : round(V0_ratio, 5),
            "V2_ratio"      : round(V2_ratio, 5),
            # Imbalance
            "V_imbalance"   : round(V_imbalance, 5),
            "I_imbalance"   : round(I_imbalance, 5),
            "Ib_Ia_ratio"   : round(Ib_Ia_ratio, 5),
            "Ic_Ia_ratio"   : round(Ic_Ia_ratio, 5),
            # Metadata
            "fault_bus"     : wf.fault_bus,
            "fault_type"    : wf.fault_type,
            "label"         : wf.label,
        }


# ==============================================================================
class DatasetGenerator:
    """
    Orchestrates fault simulation → waveform generation → feature extraction
    to produce a labeled ML training dataset.

    Parameters
    ----------
    fault_engine   : FaultEngine  – from Phase 3 setup
    wf_generator   : WaveformGenerator
    feature_extractor: FeatureExtractor
    n_buses        : int – number of buses in the system
    """

    FAULT_TYPES = [
        "Three_Phase",
        "Line_to_Ground",
        "Line_to_Line",
        "Double_Line_to_Ground",
    ]

    def __init__(self, fault_engine, wf_generator, feature_extractor,
                 n_buses: int = 5):
        self.fe    = fault_engine
        self.wg    = wf_generator
        self.fex   = feature_extractor
        self.n     = n_buses

    # ------------------------------------------------------------------
    def generate(self, Rf_values: np.ndarray = None,
                 seed: int = 42) -> pd.DataFrame:
        """
        Generate the complete labeled dataset.

        Parameters
        ----------
        Rf_values : array of fault resistance values to sweep (pu)
        seed      : random seed for reproducibility

        Returns
        -------
        pd.DataFrame with feature columns + 'label' + 'fault_type'
        """
        np.random.seed(seed)

        if Rf_values is None:
            # 40 values: 0 (bolted) → 0.1 pu, logarithmically spaced
            Rf_values = np.concatenate([
                [0.0],
                np.logspace(-3, -1, 39)
            ])

        records = []
        total   = (len(self.FAULT_TYPES) * self.n + self.n) * len(Rf_values)
        done    = 0

        print(f"\n[DatasetGenerator] Generating {total} samples...")
        print(f"  Fault types: {self.FAULT_TYPES}")
        print(f"  Buses: 1–{self.n}  |  Rf values: {len(Rf_values)}")
        print(f"  {'Sample':>8} / {total}  |  Progress")

        # ── FAULT SAMPLES ─────────────────────────────────────────────
        for ft in self.FAULT_TYPES:
            for bus_idx in range(self.n):
                for Rf in Rf_values:
                    result = self.fe.run_fault(ft, bus_idx, float(Rf))
                    wf     = self.wg.generate_fault_waveform(result)
                    feat   = self.fex.extract(wf)
                    feat['Rf_pu'] = round(float(Rf), 6)
                    records.append(feat)
                    done += 1

                if done % 100 == 0:
                    pct = done / total * 100
                    bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
                    print(f"  {done:>8} / {total}  [{bar}] {pct:.1f}%")

        # ── NO-FAULT SAMPLES ──────────────────────────────────────────
        for bus_idx in range(self.n):
            V_mag = float(np.abs(self.fe.V_pre[bus_idx]))
            for Rf in Rf_values:
                wf   = self.wg.generate_no_fault(bus_idx, V_mag)
                feat = self.fex.extract(wf)
                feat['Rf_pu'] = 0.0
                records.append(feat)
                done += 1

        print(f"\n  ✅ Generated {len(records)} samples total")

        df = pd.DataFrame(records)
        return df

    # ------------------------------------------------------------------
    def save_dataset(self, df: pd.DataFrame, path: str):
        """Save dataset CSV and print class distribution."""
        df.to_csv(path, index=False)
        print(f"\n[DatasetGenerator] Dataset saved → {path}")
        print(f"  Shape : {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\n  Class distribution:")
        counts = df.groupby(['fault_type', 'label']).size().reset_index(name='count')
        for _, row in counts.iterrows():
            bar = '█' * int(row['count'] // 20)
            print(f"    [{row['label']}] {row['fault_type']:<30}: "
                  f"{row['count']:>4} samples  {bar}")
        print(f"\n  Features ({df.shape[1]-3}):")
        feat_cols = [c for c in df.columns
                     if c not in ('fault_type', 'label', 'Rf_pu')]
        for i in range(0, len(feat_cols), 5):
            print(f"    {', '.join(feat_cols[i:i+5])}")