"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: ML Trainer  (ml/trainer.py)

  Models trained & compared:
    1. Random Forest        — ensemble of decision trees
    2. Gradient Boosting    — sequential boosted trees (XGBoost-style)
    3. MLP Neural Network   — 3-hidden-layer feedforward net
    4. Support Vector Machine (RBF kernel)
    5. K-Nearest Neighbours — baseline distance classifier

  Pipeline per model:
    → Train on X_train / y_train
    → Evaluate on X_test  / y_test
    → 5-fold stratified cross-validation
    → GridSearchCV for best model (Random Forest)
    → Save best model + scaler with joblib

  Why Random Forest is the primary model:
    • Handles mixed-scale features without extra scaling
    • Built-in feature importance ranking
    • Robust to noise and outliers in synthetic data
    • Fast inference — critical for real-time fault detection
=============================================================
"""

import numpy as np
import time, os
import joblib

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics         import accuracy_score


class ModelTrainer:
    """
    Trains, tunes, and compares 5 classifiers on the fault dataset.

    Parameters
    ----------
    data       : dict – output from Preprocessor.load_and_prepare()
    save_dir   : str  – directory to save trained models
    random_state: int – reproducibility seed
    """

    def __init__(self, data: dict, save_dir: str = "ml",
                 random_state: int = 42):
        self.X_train       = data["X_train"]
        self.X_test        = data["X_test"]
        self.y_train       = data["y_train"]
        self.y_test        = data["y_test"]
        self.feature_names = data["feature_names"]
        self.class_names   = data["class_names"]
        self.save_dir      = save_dir
        self.rs            = random_state
        self.results       = {}     # model_name → metrics dict
        self.trained       = {}     # model_name → fitted model
        os.makedirs(save_dir, exist_ok=True)

    # ==================================================================
    # DEFINE ALL 5 MODELS
    # ==================================================================
    def _build_models(self) -> dict:
        return {
            "Random_Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced",
                random_state=self.rs,
                n_jobs=-1
            ),
            "Gradient_Boosting": GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                random_state=self.rs
            ),
            "MLP_Neural_Net": MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self.rs
            ),
            "SVM_RBF": SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                random_state=self.rs
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=7,
                metric="euclidean",
                weights="distance",
                n_jobs=-1
            ),
        }

    # ==================================================================
    # TRAIN ALL MODELS
    # ==================================================================
    def train_all(self) -> dict:
        """
        Train every model, compute test accuracy + 5-fold CV score,
        store results in self.results and self.trained.
        """
        models = self._build_models()
        cv     = StratifiedKFold(n_splits=5, shuffle=True,
                                 random_state=self.rs)

        print("\n" + "─"*68)
        print(f"  {'Model':<22} {'Test Acc':>9} {'CV Mean':>9} "
              f"{'CV Std':>8} {'Time':>8}")
        print("─"*68)

        for name, model in models.items():
            t0 = time.time()

            # ── Train ─────────────────────────────────────────────────
            model.fit(self.X_train, self.y_train)

            # ── Test accuracy ─────────────────────────────────────────
            y_pred   = model.predict(self.X_test)
            test_acc = accuracy_score(self.y_test, y_pred)

            # ── 5-fold CV on full training set ───────────────────────
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=cv, scoring="accuracy", n_jobs=-1
            )
            elapsed = time.time() - t0

            self.results[name] = {
                "model"     : model,
                "y_pred"    : y_pred,
                "test_acc"  : round(test_acc * 100, 3),
                "cv_mean"   : round(cv_scores.mean() * 100, 3),
                "cv_std"    : round(cv_scores.std()  * 100, 3),
                "cv_scores" : cv_scores,
                "train_time": round(elapsed, 2),
            }
            self.trained[name] = model

            bar = "█" * int(test_acc * 20)
            print(f"  {name:<22} {test_acc*100:>8.3f}%  "
                  f"{cv_scores.mean()*100:>8.3f}%  "
                  f"±{cv_scores.std()*100:>6.3f}%  "
                  f"{elapsed:>6.2f}s  {bar}")

        print("─"*68)
        return self.results

    # ==================================================================
    # HYPERPARAMETER TUNING  (Random Forest — GridSearchCV)
    # ==================================================================
    def tune_random_forest(self) -> RandomForestClassifier:
        """
        Fine-tune Random Forest with GridSearchCV.

        Search space:
          n_estimators     : [100, 200, 300]
          max_depth        : [None, 10, 20]
          min_samples_leaf : [1, 2]
          max_features     : ['sqrt', 'log2']

        Uses 5-fold stratified CV, scoring = accuracy.
        """
        print("\n[Trainer] GridSearchCV — Random Forest Hyperparameter Tuning")
        print("  This may take ~30 seconds...")

        param_grid = {
            "n_estimators"    : [100, 200, 300],
            "max_depth"       : [None, 10, 20],
            "min_samples_leaf": [1, 2],
            "max_features"    : ["sqrt", "log2"],
        }

        cv  = StratifiedKFold(n_splits=5, shuffle=True,
                              random_state=self.rs)
        rf  = RandomForestClassifier(class_weight="balanced",
                                     random_state=self.rs, n_jobs=-1)
        gs  = GridSearchCV(rf, param_grid, cv=cv,
                           scoring="accuracy", n_jobs=-1, verbose=0)

        t0  = time.time()
        gs.fit(self.X_train, self.y_train)
        elapsed = time.time() - t0

        best_rf      = gs.best_estimator_
        best_cv_acc  = gs.best_score_ * 100
        test_acc     = best_rf.score(self.X_test, self.y_test) * 100

        print(f"\n  Best parameters  : {gs.best_params_}")
        print(f"  Best CV accuracy : {best_cv_acc:.3f}%")
        print(f"  Test accuracy    : {test_acc:.3f}%")
        print(f"  Tuning time      : {elapsed:.1f}s")

        # Update Random Forest entry with tuned model
        y_pred = best_rf.predict(self.X_test)
        self.results["Random_Forest"]["model"]    = best_rf
        self.results["Random_Forest"]["y_pred"]   = y_pred
        self.results["Random_Forest"]["test_acc"] = round(test_acc, 3)
        self.results["Random_Forest"]["best_params"] = gs.best_params_
        self.trained["Random_Forest"]             = best_rf

        return best_rf

    # ==================================================================
    # FEATURE IMPORTANCE  (Random Forest)
    # ==================================================================
    def get_feature_importance(self) -> dict:
        """
        Extract and rank feature importances from the trained Random Forest.
        Returns a dict {feature_name: importance_score} sorted descending.
        """
        rf          = self.trained["Random_Forest"]
        importances = rf.feature_importances_
        std         = np.std(
            [tree.feature_importances_ for tree in rf.estimators_], axis=0
        )
        indices     = np.argsort(importances)[::-1]

        ranked = {
            self.feature_names[i]: {
                "importance": round(float(importances[i]), 5),
                "std"       : round(float(std[i]), 5),
                "rank"      : rank + 1
            }
            for rank, i in enumerate(indices)
        }

        print(f"\n[Trainer] Top-10 Feature Importances (Random Forest):")
        print(f"  {'Rank':<5} {'Feature':<22} {'Importance':>12} {'±Std':>8}")
        print("  " + "-"*52)
        for feat, info in list(ranked.items())[:10]:
            bar = "▓" * int(info['importance'] * 200)
            print(f"  {info['rank']:<5} {feat:<22} "
                  f"{info['importance']:>12.5f}  "
                  f"±{info['std']:.5f}  {bar}")

        return ranked

    # ==================================================================
    # PICK BEST MODEL & SAVE
    # ==================================================================
    def save_best_model(self, scaler, out_dir: str = "ml") -> str:
        """
        Identify the best model by test accuracy, save it + scaler.

        Returns: name of the best model.
        """
        best_name = max(
            self.results,
            key=lambda n: self.results[n]["test_acc"]
        )
        best_acc = self.results[best_name]["test_acc"]

        model_path  = os.path.join(out_dir, "best_model.joblib")
        scaler_path = os.path.join(out_dir, "scaler.joblib")

        joblib.dump(self.trained[best_name], model_path)
        joblib.dump(scaler, scaler_path)

        print(f"\n[Trainer] Best model → '{best_name}'  "
              f"(Test Acc = {best_acc:.3f}%)")
        print(f"  Saved model  → {model_path}")
        print(f"  Saved scaler → {scaler_path}")

        # Also save summary
        import pandas as pd
        summary_rows = []
        for name, res in self.results.items():
            summary_rows.append({
                "model"      : name,
                "test_acc_%" : res["test_acc"],
                "cv_mean_%"  : res["cv_mean"],
                "cv_std_%"   : res["cv_std"],
                "train_time_s": res["train_time"],
                "is_best"    : (name == best_name),
            })
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(out_dir, "model_comparison.csv"), index=False)

        return best_name