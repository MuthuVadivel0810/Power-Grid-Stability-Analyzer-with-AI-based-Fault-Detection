# Power-Grid-Stability-Analyzer-with-AI-based-Fault-Detection
Built a Python-based power grid simulator covering Newton-Raphson load flow on an IEEE 5-bus system, symmetrical component fault analysis for 4 fault types, and a Random Forest classifier achieving 100% accuracy on 1000 synthetic waveform samples — all wrapped in a  interactive Streamlit dashboard
# ⚡ Power Grid Stability Analyzer with AI-Based Fault Detection

> A full-stack power systems simulation and machine learning project —
> from IEEE 5-bus load flow to real-time fault classification via an
> interactive Streamlit dashboard.

---

## 🎯 Project Overview

This project simulates a power grid, injects electrical faults, and uses
machine learning to automatically classify fault types in real time.

Built with **pure Python** — no MATLAB, no external power systems software.

---

## 🏗️ Architecture

```
IEEE 5-Bus Data (CSV)
       ↓
 Ybus Matrix Builder          Phase 1 — core/ybus_builder.py
       ↓
 Newton-Raphson Solver         Phase 2 — core/nr_solver.py
       ↓
 Fault Injection Engine        Phase 3 — core/fault_engine.py
 (3Φ, L-G, L-L, L-L-G)                  core/waveform_generator.py
       ↓
 Feature Extraction            Phase 3 — core/dataset_generator.py
 (FFT, RMS, THD, Sequence)
       ↓
 ML Classifier Training        Phase 4 — ml/trainer.py
 (Random Forest + 4 others)             ml/evaluator.py
       ↓
 Interactive Dashboard         Phase 5 — dashboard/app.py
```

---

## ✅ Key Results

| Metric | Value |
|---|---|
| Newton-Raphson convergence | **4 iterations** (tol = 1×10⁻⁶ pu) |
| Active power losses | **2.71% of generation** |
| ML test accuracy | **100.00%** |
| ML cross-validation | **100.00% ± 0.000%** |
| ROC-AUC (all classes) | **1.0000** |
| Zero misclassifications | **200/200 test samples** |

---

## 📁 Project Structure

```
power_grid_project/
│
├── data/
│   ├── bus_data.csv          IEEE 5-bus system data
│   └── line_data.csv         Transmission line parameters
│
├── core/
│   ├── ybus_builder.py       Bus admittance matrix (Ybus)
│   ├── nr_solver.py          Newton-Raphson load flow
│   ├── results_analyzer.py   Line flows, losses, violations
│   ├── grid_visualizer.py    NetworkX topology plots
│   ├── fault_engine.py       Symmetrical component fault analysis
│   ├── waveform_generator.py 3-phase signal synthesis
│   └── dataset_generator.py  Feature extraction + dataset builder
│
├── ml/
│   ├── preprocessor.py       Data loading, scaling, split
│   ├── trainer.py            5-model training + GridSearchCV
│   ├── evaluator.py          Confusion matrix, ROC, F1
│   ├── predictor.py          Real-time prediction API
│   ├── best_model.joblib     Saved Random Forest model
│   ├── scaler.joblib         Saved StandardScaler
│   └── fault_dataset.csv     1000-sample labeled dataset
│
├── dashboard/
│   ├── app.py                Streamlit multi-page app (5 pages)
│   ├── utils.py              Cached data loaders
│   └── charts.py             Interactive Plotly chart builders
│
├── phase1_main.py            Run Phase 1 standalone
├── phase2_main.py            Run Phase 2 standalone
├── phase3_main.py            Run Phase 3 standalone
├── phase4_main.py            Run Phase 4 standalone
├── phase5_preview.py         Generate static dashboard previews
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run each phase individually
```bash
python phase1_main.py    # Ybus + topology
python phase2_main.py    # NR load flow
python phase3_main.py    # Fault injection + dataset
python phase4_main.py    # ML training + evaluation
```

### 3. Launch the dashboard
```bash
streamlit run dashboard/app.py
```
Open http://localhost:8501 in your browser.

---

## 🧠 Technical Concepts Implemented

### Power Systems
- **Bus Admittance Matrix** (Ybus) from π-model transmission lines
- **Newton-Raphson Load Flow** with 7×7 Jacobian (H, N, J, L submatrices)
- **Symmetrical Component Theory** — Z1, Z2, Z0 sequence networks
- **4 Fault Types** via Thevenin equivalent circuits
- **Post-fault voltage propagation** using superposition principle

### Signal Processing
- **FFT** harmonic extraction (2nd, 3rd, 5th harmonics)
- **Total Harmonic Distortion (THD)** calculation
- **RMS** computation over windowed time segments
- **Sequence component ratios** (V0/V1, V2/V1)

### Machine Learning
- **Random Forest** (200 trees, GridSearchCV tuned)
- **5-fold stratified cross-validation**
- **Feature importance** with standard deviation
- **ROC-AUC**, precision-recall, confusion matrix evaluation
- **32 engineered features** including 3 domain-derived features

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Overview | Project pipeline, grid topology, fault reference |
| 🔌 Load Flow | NR convergence, voltage profile, line flows |
| ⚡ Fault Simulator | Inject any fault → waveform → ML prediction |
| 🤖 ML Analytics | Model comparison, ROC, feature importance |
| 📊 Dataset Explorer | Scatter plots, distributions, download CSV |

---

## 🛠️ Technologies Used

`Python` `NumPy` `Pandas` `Matplotlib` `SciPy` `scikit-learn`
`NetworkX` `Streamlit` `Plotly` `joblib`

---

*Final Year Project — Electrical & Electronics Engineering*
*Power Grid Stability Analyzer with AI-Based Fault Detection*
