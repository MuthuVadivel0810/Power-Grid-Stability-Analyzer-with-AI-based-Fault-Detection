"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: Predictor  (ml/predictor.py)

  What this does:
    - Loads the saved best_model.joblib + scaler.joblib
    - Provides a clean predict() API:
        Input  : raw waveform (Waveform object) OR feature dict
        Output : fault_type (str), confidence (%), all class probs

  This is what the Streamlit dashboard (Phase 5) will call
  in real-time when the user clicks "Inject Fault & Predict".
=============================================================
"""

import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ML.Preprocessor import CLASS_NAMES, NON_FEATURE_COLS


FEATURE_ORDER = [
    "Va_rms_pre", "Va_rms_flt", "Vb_rms_flt", "Vc_rms_flt",
    "Va_sag",     "Vb_sag",     "Vc_sag",
    "Ia_rms_flt", "Ib_rms_flt", "Ic_rms_flt",
    "Ia_peak",    "Ib_peak",    "Ic_peak",    "I_max_peak",
    "Va_H2",      "Va_H3",      "Va_H5",
    "Ia_H2",      "Ia_H3",      "Ib_H2",      "Ib_H3",
    "Va_thd",     "Ia_thd",
    "V0_ratio",   "V2_ratio",
    "V_imbalance","I_imbalance",
    "Ib_Ia_ratio","Ic_Ia_ratio",
    "fault_bus",
    # Engineered features
    "sag_asymmetry", "current_ratio", "harmonic_index",
]


class FaultPredictor:
    """
    Real-time fault classification from a feature vector.

    Usage
    -----
    predictor = FaultPredictor("ml/best_model.joblib",
                               "ml/scaler.joblib")
    result = predictor.predict_from_dict(feature_dict)
    print(result)
    """

    def __init__(self, model_path: str, scaler_path: str):
        self.model  = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"[FaultPredictor] Model loaded  → {model_path}")
        print(f"[FaultPredictor] Scaler loaded → {scaler_path}")

    # ==================================================================
    def predict_from_dict(self, feature_dict: dict) -> dict:
        """
        Predict fault type from a feature dictionary.

        Parameters
        ----------
        feature_dict : dict with keys matching FEATURE_ORDER
                       (extra keys like fault_type/label are ignored)

        Returns
        -------
        dict:
          predicted_label   : int
          predicted_class   : str
          confidence        : float (0–100%)
          all_probabilities : dict {class_name: probability}
          is_fault          : bool
        """
        # ── Add engineered features if not already present ─────────────
        fd = feature_dict.copy()
        if "sag_asymmetry" not in fd:
            fd["sag_asymmetry"] = (
                abs(fd.get("Va_sag", 0) - fd.get("Vb_sag", 0)) +
                abs(fd.get("Vb_sag", 0) - fd.get("Vc_sag", 0))
            )
        if "current_ratio" not in fd:
            fd["current_ratio"] = fd.get("I_max_peak", 0) / (
                fd.get("Ia_rms_flt", 1e-9) + 1e-9)
        if "harmonic_index" not in fd:
            fd["harmonic_index"] = (fd.get("Va_H2", 0) +
                                    fd.get("Va_H3", 0) +
                                    fd.get("Va_H5", 0))

        # ── Build feature vector in correct order ──────────────────────
        x = np.array([[fd.get(f, 0.0) for f in FEATURE_ORDER]])

        # ── Scale ──────────────────────────────────────────────────────
        x_scaled = self.scaler.transform(x)

        # ── Predict ────────────────────────────────────────────────────
        label = int(self.model.predict(x_scaled)[0])
        probs = self.model.predict_proba(x_scaled)[0]

        return {
            "predicted_label"  : label,
            "predicted_class"  : CLASS_NAMES[label],
            "confidence"       : round(float(probs[label]) * 100, 2),
            "all_probabilities": {
                CLASS_NAMES[i]: round(float(p) * 100, 2)
                for i, p in enumerate(probs)
            },
            "is_fault"         : label != 0,
        }

    # ==================================================================
    def predict_from_waveform(self, waveform, extractor) -> dict:
        """
        Full pipeline: Waveform object → features → prediction.

        Parameters
        ----------
        waveform  : Waveform object from WaveformGenerator
        extractor : FeatureExtractor instance
        """
        feature_dict = extractor.extract(waveform)
        return self.predict_from_dict(feature_dict)

    # ==================================================================
    def batch_predict(self, feature_df) -> list:
        """
        Predict fault type for multiple samples (DataFrame input).

        Returns list of result dicts.
        """
        results = []
        for _, row in feature_df.iterrows():
            results.append(self.predict_from_dict(row.to_dict()))
        return results

    # ==================================================================
    def print_prediction(self, result: dict):
        """Pretty-print a single prediction result."""
        print(f"\n{'─'*45}")
        print(f"  ⚡ FAULT PREDICTION RESULT")
        print(f"{'─'*45}")
        print(f"  Predicted Class  : {result['predicted_class']}")
        print(f"  Confidence       : {result['confidence']:.2f}%")
        print(f"  Is Fault?        : {'YES ⚠' if result['is_fault'] else 'NO ✓'}")
        print(f"\n  Class Probabilities:")
        for cls, prob in sorted(result['all_probabilities'].items(),
                                key=lambda x: -x[1]):
            bar = "█" * int(prob / 5)
            print(f"    {cls:<16}: {prob:>6.2f}%  {bar}")
        print(f"{'─'*45}")