"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: ML Preprocessor  (ml/preprocessor.py)

  What this does:
    1. Loads  ml/fault_dataset.csv
    2. Drops non-feature columns, handles any NaN
    3. Adds 3 engineered features on top of the 29 raw ones:
         • sag_asymmetry   : |Va_sag - Vb_sag| + |Vb_sag - Vc_sag|
         • current_ratio   : I_max_peak / (Ia_rms_flt + ε)
         • harmonic_index  : Va_H2 + Va_H3 + Va_H5   (composite)
    4. Splits into train (80%) / test (20%), stratified by class
    5. Applies StandardScaler  (fit on train, transform both)
    6. Returns ready-to-use arrays + metadata

  Why StandardScaler?
    Features like I_max_peak (0.3–1.1 pu) and Va_H2 (0.001–0.3)
    are on very different scales. Scaling prevents large-valued
    features from dominating distance-based / gradient models.
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
import joblib, os

CLASS_NAMES = {
    0: "No Fault",
    1: "3-Phase",
    2: "Line-Ground",
    3: "Line-Line",
    4: "DLine-Ground",
}

# Columns that are NOT features
NON_FEATURE_COLS = ['fault_type', 'label', 'Rf_pu']


class Preprocessor:
    """
    Loads, engineers, splits, and scales the fault dataset.

    Parameters
    ----------
    csv_path   : str   – path to fault_dataset.csv
    test_size  : float – fraction held out for testing  (default 0.20)
    random_state: int  – reproducibility seed
    """

    def __init__(self, csv_path: str,
                 test_size: float = 0.20,
                 random_state: int = 42):
        self.csv_path     = csv_path
        self.test_size    = test_size
        self.random_state = random_state
        self.scaler       = StandardScaler()
        self.feature_cols = []

    # ==================================================================
    def load_and_prepare(self) -> dict:
        """
        Full preprocessing pipeline.

        Returns
        -------
        dict with keys:
          X_train, X_test, y_train, y_test  (numpy arrays)
          X_train_raw, X_test_raw           (unscaled, for inspection)
          feature_names                     (list of feature column names)
          class_names                       (dict  label → string)
          df                                (full raw DataFrame)
        """
        # ── Load ──────────────────────────────────────────────────────
        df = pd.read_csv(self.csv_path)
        print(f"[Preprocessor] Loaded {len(df)} samples × {df.shape[1]} cols")

        # ── Drop NaN ──────────────────────────────────────────────────
        before = len(df)
        df.dropna(inplace=True)
        if len(df) < before:
            print(f"  Dropped {before - len(df)} rows with NaN values")

        # ── Feature engineering ───────────────────────────────────────
        df['sag_asymmetry']  = (
            (df['Va_sag'] - df['Vb_sag']).abs() +
            (df['Vb_sag'] - df['Vc_sag']).abs()
        )
        df['current_ratio']  = df['I_max_peak'] / (df['Ia_rms_flt'] + 1e-9)
        df['harmonic_index'] = df['Va_H2'] + df['Va_H3'] + df['Va_H5']

        print(f"  Added 3 engineered features: "
              f"sag_asymmetry, current_ratio, harmonic_index")

        # ── Feature / target split ────────────────────────────────────
        drop_cols = [c for c in NON_FEATURE_COLS if c in df.columns]
        X = df.drop(columns=drop_cols).values.astype(float)
        y = df['label'].values.astype(int)

        self.feature_cols = [c for c in df.columns if c not in drop_cols]

        print(f"  Feature matrix : {X.shape[0]} × {X.shape[1]}")
        print(f"  Target classes : {np.unique(y)}")

        # ── Train / test split (stratified) ───────────────────────────
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )
        print(f"  Train: {len(y_train)} samples  |  "
              f"Test: {len(y_test)} samples  (stratified)")

        # ── Scaling ───────────────────────────────────────────────────
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test  = self.scaler.transform(X_test_raw)
        print(f"  StandardScaler fit on train set ✓")

        # ── Class balance check ───────────────────────────────────────
        for label, name in CLASS_NAMES.items():
            count = np.sum(y_train == label)
            print(f"    Class {label} ({name:<14}): "
                  f"{count} train / {np.sum(y_test == label)} test")

        return {
            "X_train"      : X_train,
            "X_test"       : X_test,
            "X_train_raw"  : X_train_raw,
            "X_test_raw"   : X_test_raw,
            "y_train"      : y_train,
            "y_test"       : y_test,
            "feature_names": self.feature_cols,
            "class_names"  : CLASS_NAMES,
            "df"           : df,
        }

    # ------------------------------------------------------------------
    def save_scaler(self, path: str):
        joblib.dump(self.scaler, path)
        print(f"  Scaler saved → {path}")

    def load_scaler(self, path: str):
        self.scaler = joblib.load(path)

    def transform_new(self, X_new: np.ndarray) -> np.ndarray:
        """Scale a new sample using the fitted scaler."""
        return self.scaler.transform(X_new)