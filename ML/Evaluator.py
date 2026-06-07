"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: ML Evaluator  (ml/evaluator.py)

  What this does:
    1. Full classification report  (precision, recall, F1 per class)
    2. Confusion matrix            (raw counts + normalised)
    3. Per-class ROC-AUC           (One-vs-Rest, probability output)
    4. Misclassification analysis  (which fault types are confused)
    5. Learning curve              (train vs CV score vs training size)
    6. All metrics bundled in a dict for the plotter
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    accuracy_score, f1_score
)
from sklearn.model_selection import learning_curve, StratifiedKFold


CLASS_NAMES = {
    0: "No Fault",
    1: "3-Phase",
    2: "Line-Ground",
    3: "Line-Line",
    4: "DLine-Ground",
}


class Evaluator:
    """
    Comprehensive evaluation of a trained fault classifier.

    Parameters
    ----------
    model      : fitted sklearn estimator
    X_test     : np.ndarray  – scaled test features
    y_test     : np.ndarray  – true labels
    X_train    : np.ndarray  – scaled train features (for learning curve)
    y_train    : np.ndarray  – train labels
    """

    def __init__(self, model, X_test, y_test, X_train, y_train):
        self.model   = model
        self.X_test  = X_test
        self.y_test  = y_test
        self.X_train = X_train
        self.y_train = y_train

        self.y_pred  = model.predict(X_test)
        self.classes = sorted(CLASS_NAMES.keys())
        self.n_cls   = len(self.classes)

        # Probability outputs (for ROC / PR curves)
        if hasattr(model, "predict_proba"):
            self.y_prob = model.predict_proba(X_test)
        else:
            self.y_prob = None

    # ==================================================================
    # 1.  CLASSIFICATION REPORT
    # ==================================================================
    def classification_report_df(self) -> pd.DataFrame:
        """Return sklearn classification report as a clean DataFrame."""
        report = classification_report(
            self.y_test, self.y_pred,
            target_names=[CLASS_NAMES[c] for c in self.classes],
            output_dict=True
        )
        df = pd.DataFrame(report).T
        df = df.drop(index=["accuracy"], errors="ignore")
        df = df.rename(columns={
            "precision": "Precision",
            "recall"   : "Recall",
            "f1-score" : "F1-Score",
            "support"  : "Support"
        })
        df[["Precision","Recall","F1-Score"]] = \
            df[["Precision","Recall","F1-Score"]].round(4)
        return df

    # ==================================================================
    # 2.  CONFUSION MATRIX
    # ==================================================================
    def confusion_matrices(self) -> tuple:
        """
        Returns (cm_raw, cm_norm):
          cm_raw  : int array   — absolute counts
          cm_norm : float array — row-normalised (recall per class)
        """
        cm_raw  = confusion_matrix(self.y_test, self.y_pred,
                                   labels=self.classes)
        cm_norm = cm_raw.astype(float) / (cm_raw.sum(axis=1, keepdims=True) + 1e-9)
        return cm_raw, cm_norm

    # ==================================================================
    # 3.  ROC-AUC (One-vs-Rest)
    # ==================================================================
    def roc_data(self) -> dict:
        """
        Compute ROC curve + AUC for each class (One-vs-Rest).
        Returns dict: class_label → {fpr, tpr, auc}
        """
        if self.y_prob is None:
            return {}

        roc = {}
        for i, cls in enumerate(self.classes):
            y_bin = (self.y_test == cls).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, self.y_prob[:, i])
            auc = roc_auc_score(y_bin, self.y_prob[:, i])
            roc[cls] = {"fpr": fpr, "tpr": tpr, "auc": round(auc, 5)}
        return roc

    # ==================================================================
    # 4.  PRECISION-RECALL DATA
    # ==================================================================
    def pr_data(self) -> dict:
        """Precision-Recall curves for each class."""
        if self.y_prob is None:
            return {}
        pr = {}
        for i, cls in enumerate(self.classes):
            y_bin = (self.y_test == cls).astype(int)
            prec, rec, _ = precision_recall_curve(y_bin, self.y_prob[:, i])
            ap = average_precision_score(y_bin, self.y_prob[:, i])
            pr[cls] = {"precision": prec, "recall": rec, "ap": round(ap, 5)}
        return pr

    # ==================================================================
    # 5.  LEARNING CURVE
    # ==================================================================
    def learning_curve_data(self) -> dict:
        """
        Train model at different training-set sizes (10%…100%)
        to check for overfitting / underfitting.
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        sizes = np.linspace(0.10, 1.0, 10)

        train_sizes, train_scores, val_scores = learning_curve(
            self.model,
            self.X_train, self.y_train,
            train_sizes=sizes, cv=cv,
            scoring="accuracy", n_jobs=-1
        )
        return {
            "train_sizes"  : train_sizes,
            "train_mean"   : train_scores.mean(axis=1),
            "train_std"    : train_scores.std(axis=1),
            "val_mean"     : val_scores.mean(axis=1),
            "val_std"      : val_scores.std(axis=1),
        }

    # ==================================================================
    # 6.  MISCLASSIFICATION ANALYSIS
    # ==================================================================
    def misclassification_analysis(self) -> pd.DataFrame:
        """
        Find every misclassified test sample and record:
          true label, predicted label, max confidence
        """
        wrong_mask = (self.y_pred != self.y_test)
        if not np.any(wrong_mask):
            print("  Perfect classification — zero misclassifications!")
            return pd.DataFrame()

        rows = []
        for idx in np.where(wrong_mask)[0]:
            conf = (float(np.max(self.y_prob[idx]))
                    if self.y_prob is not None else None)
            rows.append({
                "True"       : CLASS_NAMES[self.y_test[idx]],
                "Predicted"  : CLASS_NAMES[self.y_pred[idx]],
                "Confidence" : round(conf, 4) if conf else "N/A",
            })

        df = pd.DataFrame(rows)
        summary = df.groupby(["True", "Predicted"]).size().reset_index(name="Count")
        return summary

    # ==================================================================
    # 7.  FULL PRINT REPORT
    # ==================================================================
    def print_full_report(self, model_name: str = "Best Model"):
        acc    = accuracy_score(self.y_test, self.y_pred)
        f1_mac = f1_score(self.y_test, self.y_pred, average="macro")

        print(f"\n{'='*65}")
        print(f"  MODEL EVALUATION REPORT  —  {model_name}")
        print(f"{'='*65}")
        print(f"  Overall Test Accuracy : {acc*100:.4f}%")
        print(f"  Macro F1-Score        : {f1_mac:.4f}")

        print(f"\n  Per-Class Metrics:")
        cr_df = self.classification_report_df()
        print(cr_df.to_string())

        # ROC AUC
        roc = self.roc_data()
        if roc:
            print(f"\n  Per-Class ROC-AUC:")
            for cls, d in roc.items():
                bar = "█" * int(d['auc'] * 20)
                print(f"    [{cls}] {CLASS_NAMES[cls]:<15} : "
                      f"AUC = {d['auc']:.5f}  {bar}")

        # Confusion matrix
        cm_raw, _ = self.confusion_matrices()
        print(f"\n  Confusion Matrix (rows=True, cols=Predicted):")
        lbl = [f"{CLASS_NAMES[c]:>12}" for c in self.classes]
        print("  " + " ".join(lbl))
        for i, row in enumerate(cm_raw):
            print(f"  {CLASS_NAMES[self.classes[i]]:<14}"
                  + " ".join(f"{v:>12}" for v in row))

        # Misclassification
        misc = self.misclassification_analysis()
        if len(misc) > 0:
            print(f"\n  Misclassification Summary:")
            print(misc.to_string(index=False))
        else:
            print(f"\n  ✅ Zero misclassifications on test set!")

        print(f"{'='*65}")

    # ==================================================================
    # 8.  BUNDLE ALL METRICS
    # ==================================================================
    def get_all_metrics(self) -> dict:
        """Return everything in one dict for the plotter."""
        cm_raw, cm_norm = self.confusion_matrices()
        return {
            "accuracy"              : accuracy_score(self.y_test, self.y_pred),
            "f1_macro"              : f1_score(self.y_test, self.y_pred, average="macro"),
            "classification_report" : self.classification_report_df(),
            "cm_raw"                : cm_raw,
            "cm_norm"               : cm_norm,
            "roc"                   : self.roc_data(),
            "pr"                    : self.pr_data(),
            "learning_curve"        : self.learning_curve_data(),
            "misclassifications"    : self.misclassification_analysis(),
            "y_pred"                : self.y_pred,
            "y_test"                : self.y_test,
            "y_prob"                : self.y_prob,
            "class_names"           : CLASS_NAMES,
        }