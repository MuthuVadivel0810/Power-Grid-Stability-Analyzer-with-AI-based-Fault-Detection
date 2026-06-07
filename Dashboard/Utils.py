"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: Dashboard Utilities  (dashboard/utils.py)

  Shared cached loaders used by every Streamlit page:
    • load_grid_data()      – bus & line DataFrames
    • load_nr_results()     – full Newton-Raphson solution
    • load_fault_engine()   – FaultEngine ready to fire
    • load_predictor()      – trained ML model + scaler
    • load_dataset()        – fault_dataset.csv for analytics

  All functions decorated with @st.cache_resource or
  @st.cache_data so they run only once per session.
=============================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import streamlit as st

from Core.Ybus_builder       import YbusBuilder
from Core.NR_solver          import NRSolver
from Core.Fault_Engine       import FaultEngine
from Core.Waveform_Generator import WaveformGenerator
from Core.Dataset_Generator  import FeatureExtractor
from ML.Predictor            import FaultPredictor

BASE   = os.path.join(os.path.dirname(__file__), "..")
BUS_F  = os.path.join(BASE, "data", "bus_data.csv")
LINE_F = os.path.join(BASE, "data", "line_data.csv")
MODEL  = os.path.join(BASE, "ml",   "best_model.joblib")
SCALER = os.path.join(BASE, "ml",   "scaler.joblib")
DATA_F = os.path.join(BASE, "ml",   "fault_dataset.csv")

FAULT_TYPES = [
    "No_Fault",
    "Three_Phase",
    "Line_to_Ground",
    "Line_to_Line",
    "Double_Line_to_Ground",
]

FAULT_COLORS = {
    "No_Fault"             : "#2ECC71",
    "Three_Phase"          : "#E74C3C",
    "Line_to_Ground"       : "#3498DB",
    "Line_to_Line"         : "#F39C12",
    "Double_Line_to_Ground": "#9B59B6",
}

CLASS_NAMES = {
    0: "No Fault",
    1: "Three Phase",
    2: "Line to Ground",
    3: "Line to Line",
    4: "Double Line to Ground",
}

# ── Plotly dark theme defaults ─────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1A1A2E",
    plot_bgcolor ="#16213E",
    font         =dict(color="#ECEFF4", family="monospace"),
    margin       =dict(l=50, r=30, t=50, b=50),
)


@st.cache_resource
def load_grid_system():
    """Build Ybus + run NR solver. Cached for full session."""
    bus_data  = pd.read_csv(BUS_F)
    line_data = pd.read_csv(LINE_F)

    builder = YbusBuilder(BUS_F, LINE_F)
    Ybus    = builder.build()

    solver    = NRSolver(BUS_F, LINE_F, Ybus, tolerance=1e-6)
    nr_result = solver.solve()

    V_prefault = np.array([
        nr_result["V"][i] * np.exp(1j * nr_result["theta_rad"][i])
        for i in range(len(nr_result["V"]))
    ])

    fault_engine = FaultEngine(Ybus, V_prefault, Z0_factor=3.0)

    return {
        "bus_data"    : bus_data,
        "line_data"   : line_data,
        "Ybus"        : Ybus,
        "nr_result"   : nr_result,
        "V_prefault"  : V_prefault,
        "fault_engine": fault_engine,
    }


@st.cache_resource
def load_predictor():
    """Load trained ML model + scaler. Cached for full session."""
    return FaultPredictor(MODEL, SCALER)


@st.cache_resource
def load_waveform_tools():
    """Create waveform generator + feature extractor."""
    wg  = WaveformGenerator(fs=6400, duration=0.1,
                             fault_start=0.04, fault_dur=0.04,
                             freq=50.0, noise_std=0.005)
    fex = FeatureExtractor(fs=6400, freq=50.0)
    return wg, fex


@st.cache_data
def load_dataset():
    """Load fault dataset for analytics page."""
    return pd.read_csv(DATA_F)