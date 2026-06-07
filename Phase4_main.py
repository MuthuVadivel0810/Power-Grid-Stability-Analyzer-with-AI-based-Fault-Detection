"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Phase 4 Runner : phase4_main.py

  Pipeline:
    1.  Load & preprocess  fault_dataset.csv
    2.  Train 5 ML models  (RF, GB, MLP, SVM, KNN)
    3.  Tune best model    (GridSearchCV on Random Forest)
    4.  Evaluate best model (confusion matrix, ROC, F1)
    5.  Feature importance  analysis
    6.  Learning curve      (overfit / underfit check)
    7.  Save best model + scaler  → ml/best_model.joblib
    8.  Live prediction demo      (3 sample waveforms)
    9.  Generate all 7 plots + combined dashboard
   10.  Export all metrics to CSV

  Run with:  python phase4_main.py
=============================================================
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from ML.Preprocessor  import Preprocessor
from ML.Trainer       import ModelTrainer
from ML.Evaluator     import Evaluator
from ML.Predictor     import FaultPredictor
from ML.Phase4_Plotter import Phase4Plotter

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET   = "ml/fault_dataset.csv"
ML_DIR    = "ml"
OUT_DIR   = "outputs/phase4"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("\n" + "═"*68)
    print("  ⚡  POWER GRID STABILITY ANALYZER  —  Phase 4")
    print("      ML Fault Classifier Training & Evaluation")
    print("═"*68)

    # ══════════════════════════════════════════════════════════════════
    # STEP 1 — Load & preprocess dataset
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 1] Loading and preprocessing dataset...")
    prep = Preprocessor(DATASET, test_size=0.20, random_state=42)
    data = prep.load_and_prepare()
    prep.save_scaler(f"{ML_DIR}/scaler.joblib")

    print(f"\n  Dataset ready:")
    print(f"    Train : {data['X_train'].shape[0]} samples × "
          f"{data['X_train'].shape[1]} features")
    print(f"    Test  : {data['X_test'].shape[0]} samples × "
          f"{data['X_test'].shape[1]} features")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2 — Train all 5 models
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 2] Training all 5 classifiers...")
    trainer = ModelTrainer(data, save_dir=ML_DIR)
    results = trainer.train_all()

    # ══════════════════════════════════════════════════════════════════
    # STEP 3 — Hyperparameter tuning (Random Forest)
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 3] Hyperparameter tuning — Random Forest...")
    best_rf = trainer.tune_random_forest()

    # ══════════════════════════════════════════════════════════════════
    # STEP 4 — Pick & save best model
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 4] Selecting and saving best model...")
    best_name = trainer.save_best_model(prep.scaler, out_dir=ML_DIR)

    # ══════════════════════════════════════════════════════════════════
    # STEP 5 — Feature importance
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 5] Computing feature importances...")
    importance_dict = trainer.get_feature_importance()

    # Save importance CSV
    imp_rows = [{"feature": f, **v} for f, v in importance_dict.items()]
    pd.DataFrame(imp_rows).to_csv(
        f"{OUT_DIR}/feature_importance.csv", index=False)
    print(f"  Saved: {OUT_DIR}/feature_importance.csv")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6 — Full model evaluation
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 6] Full evaluation of best model...")
    best_model = trainer.trained[best_name]
    evaluator  = Evaluator(
        best_model,
        data["X_test"],  data["y_test"],
        data["X_train"], data["y_train"]
    )
    evaluator.print_full_report(model_name=best_name)
    metrics = evaluator.get_all_metrics()

    # Save classification report
    metrics["classification_report"].to_csv(
        f"{OUT_DIR}/classification_report.csv")
    print(f"  Saved: {OUT_DIR}/classification_report.csv")

    # Save confusion matrix
    cm_df = pd.DataFrame(
        metrics["cm_raw"],
        index  =[v for v in data["class_names"].values()],
        columns=[v for v in data["class_names"].values()]
    )
    cm_df.to_csv(f"{OUT_DIR}/confusion_matrix.csv")
    print(f"  Saved: {OUT_DIR}/confusion_matrix.csv")

    # ══════════════════════════════════════════════════════════════════
    # STEP 7 — Model comparison summary
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 7] Model comparison summary:")
    print(f"\n  {'Model':<24} {'Test Acc':>10} {'CV Mean':>10}"
          f" {'CV Std':>8} {'Best?':>7}")
    print("  " + "─"*64)
    for name, res in sorted(results.items(),
                             key=lambda x: -x[1]["test_acc"]):
        star = "  ★" if name == best_name else ""
        print(f"  {name:<24} {res['test_acc']:>9.3f}%"
              f" {res['cv_mean']:>9.3f}%"
              f" ±{res['cv_std']:>5.3f}%{star}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 8 — Live prediction demo
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 8] Live prediction demo — 3 test samples...")
    predictor = FaultPredictor(
        f"{ML_DIR}/best_model.joblib",
        f"{ML_DIR}/scaler.joblib"
    )

    # Pick 3 random samples from test set (one per fault class)
    demo_labels = [1, 2, 3]   # 3-Phase, L-G, L-L
    label_names  = data["class_names"]

    for target_label in demo_labels:
        # find a test sample of this class
        idx    = np.where(data["y_test"] == target_label)[0][0]
        x_raw  = data["X_test_raw"][idx]
        true_l = data["y_test"][idx]

        # Build feature dict from raw array
        feat_dict = dict(zip(data["feature_names"], x_raw))

        print(f"\n  → True label: [{true_l}] {label_names[true_l]}")
        result = predictor.predict_from_dict(feat_dict)
        predictor.print_prediction(result)

    # ══════════════════════════════════════════════════════════════════
    # STEP 9 — Generate all plots
    # ══════════════════════════════════════════════════════════════════
    print("\n[STEP 9] Generating all Phase 4 plots...")
    plotter = Phase4Plotter()

    plotter.plot_model_comparison(
        results,
        save_path=f"{OUT_DIR}/model_comparison.png")

    plotter.plot_confusion_matrix(
        metrics["cm_norm"], metrics["cm_raw"],
        save_path=f"{OUT_DIR}/confusion_matrix.png")

    plotter.plot_roc_curves(
        metrics["roc"],
        save_path=f"{OUT_DIR}/roc_curves.png")

    plotter.plot_feature_importance(
        importance_dict,
        save_path=f"{OUT_DIR}/feature_importance.png")

    plotter.plot_learning_curve(
        metrics["learning_curve"],
        save_path=f"{OUT_DIR}/learning_curve.png")

    plotter.plot_pr_curves(
        metrics["pr"],
        save_path=f"{OUT_DIR}/pr_curves.png")

    # ── Full dashboard ────────────────────────────────────────────────
    print("\n[STEP 10] Generating combined dashboard...")
    plotter.plot_dashboard(
        results, metrics, importance_dict, best_name,
        save_path=f"{OUT_DIR}/dashboard_phase4.png")

    # ══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    acc    = metrics["accuracy"] * 100
    f1     = metrics["f1_macro"]
    roc_d  = metrics["roc"]
    aucs   = [d["auc"] for d in roc_d.values()]

    print("\n" + "═"*68)
    print("  ✅  Phase 4 COMPLETE — ML Classifier Ready")
    print(f"\n  Best Model       : {best_name}")
    print(f"  Test Accuracy    : {acc:.4f}%")
    print(f"  Macro F1-Score   : {f1:.4f}")
    print(f"  Mean ROC-AUC     : {np.mean(aucs):.4f}")
    print(f"  Min class AUC    : {np.min(aucs):.4f}")
    print(f"\n  Saved artifacts:")
    print(f"    ml/best_model.joblib          ← trained model")
    print(f"    ml/scaler.joblib              ← feature scaler")
    print(f"    ml/model_comparison.csv")
    print(f"    outputs/phase4/dashboard_phase4.png")
    print(f"    outputs/phase4/confusion_matrix.png")
    print(f"    outputs/phase4/roc_curves.png")
    print(f"    outputs/phase4/feature_importance.png")
    print(f"\n  Next → Phase 5: Interactive Streamlit Dashboard")
    print("═"*68 + "\n")


if __name__ == "__main__":
    main()