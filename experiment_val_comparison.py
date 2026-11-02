import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss
)

from pulseguard.models.classifier import PulseGuardClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ValExperiment")

def run_validation_experiments():
    features_csv = "data/features_nfi.csv"
    val_report_path = "data/validation_experiments_report.json"
    models_dir = "models_saved"
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(features_csv)
    feature_cols = PulseGuardClassifier.FEATURE_NAMES

    # Strictly extract train and val splits (TEST AND TEST_UNSEEN ARE NOT ACCESSED)
    train_df = df[(df["split"] == "train") & (df["extraction_success"] == True)].copy()
    val_df = df[(df["split"] == "val") & (df["extraction_success"] == True)].copy()

    logger.info(f"Loaded training samples: {len(train_df)}")
    logger.info(f"Loaded validation samples: {len(val_df)}")

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values.astype(int)  # 0=Manipulated, 1=Genuine

    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values.astype(int)

    # 1. Define candidate architectures
    # All pipelines use SimpleImputer and StandardScaler (or trees with/without scaler)
    # Both LR and RF use balanced class weights, consistent random_state=42
    candidates = {
        "svm_rbf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
        ]),
        "svm_calibrated_cv": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", CalibratedClassifierCV(
                estimator=SVC(kernel="rbf", class_weight="balanced", random_state=42),
                method="sigmoid",
                cv=5
            ))
        ]),
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(n_estimators=150, max_depth=5, class_weight="balanced", random_state=42))
        ])
    }

    results = {}
    saved_artifacts = {}

    for name, pipe in candidates.items():
        logger.info(f"Fitting candidate: {name} on training data exclusively...")
        pipe.fit(X_train, y_train)

        # Save candidate artifact under distinct new filename
        artifact_path = os.path.join(models_dir, f"candidate_{name}_nfi.joblib")
        joblib.dump(pipe, artifact_path)
        saved_artifacts[name] = artifact_path

        # Predict on validation split
        y_val_pred_default = pipe.predict(X_val)
        y_val_prob_genuine = pipe.predict_proba(X_val)[:, 1]  # P(Genuine)
        y_val_prob_manip = 1.0 - y_val_prob_genuine           # P(Manipulated)

        # Default threshold (0.5 on P(Genuine)) metrics
        cm_def = confusion_matrix(y_val, y_val_pred_default, labels=[0, 1])
        tn, fp, fn, tp = cm_def.ravel()

        # Calibration diagnostics (evaluating predicted P(Genuine) vs true y_val)
        brier = float(brier_score_loss(y_val, y_val_prob_genuine))
        logloss = float(log_loss(y_val, y_val_prob_genuine))

        # Reliability curve (5 bins due to N=92)
        prob_true, prob_pred = calibration_curve(y_val, y_val_prob_genuine, n_bins=5, strategy="uniform")

        # Threshold sweep on P(Manipulated)
        # Decision rule: Predict Manipulated if P(Manipulated) >= T, else Genuine
        # Threshold T ranging from 0.10 to 0.90 in increments of 0.05
        thresholds = np.linspace(0.10, 0.90, 17)
        sweep_data = []

        for t in thresholds:
            t = float(round(t, 2))
            # If P(Manipulated) >= t -> predicted class is 0 (Manipulated)
            # Else -> predicted class is 1 (Genuine)
            pred_t = np.where(y_val_prob_manip >= t, 0, 1)

            cm_t = confusion_matrix(y_val, pred_t, labels=[0, 1])
            tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

            total_genuine = int((y_val == 1).sum())       # 54
            total_manip = int((y_val == 0).sum())         # 38

            # Genuine falsely flagged as manipulated = fn_t (y_true=1, pred=0)
            fpr_genuine = float(fn_t / total_genuine)

            # Manipulated correctly detected = tn_t (y_true=0, pred=0)
            manip_recall = float(tn_t / total_manip)

            # Manipulated precision: of all predicted manipulated (tn_t + fn_t), how many are truly manipulated
            total_pred_manip = tn_t + fn_t
            manip_precision = float(tn_t / total_pred_manip) if total_pred_manip > 0 else 0.0
            manip_f1 = float(2 * manip_precision * manip_recall / (manip_precision + manip_recall)) if (manip_precision + manip_recall) > 0 else 0.0

            # Coverage / decision breakdown
            sweep_data.append({
                "threshold_p_manip": t,
                "fpr_on_genuine": round(fpr_genuine, 4),
                "manipulated_recall": round(manip_recall, 4),
                "manipulated_precision": round(manip_precision, 4),
                "manipulated_f1": round(manip_f1, 4),
                "genuine_flagged_as_manip": int(fn_t),
                "manip_detected": int(tn_t),
                "total_pred_manip": int(total_pred_manip),
                "confusion_matrix": cm_t.tolist()
            })

        # End-to-end vs Model-only evaluation
        # Separate Reliable vs Inconclusive subsets on validation
        val_rel_mask = (val_df["signal_quality_status"] == "Reliable").values
        val_inc_mask = (val_df["signal_quality_status"] == "Inconclusive").values

        # On reliable subset
        cm_rel = confusion_matrix(y_val[val_rel_mask], y_val_pred_default[val_rel_mask], labels=[0, 1])
        tn_r, fp_r, fn_r, tp_r = cm_rel.ravel()
        auc_rel = float(roc_auc_score(y_val[val_rel_mask], y_val_prob_genuine[val_rel_mask]))

        # On inconclusive subset
        cm_inc = confusion_matrix(y_val[val_inc_mask], y_val_pred_default[val_inc_mask], labels=[0, 1])
        tn_i, fp_i, fn_i, tp_i = cm_inc.ravel()
        auc_inc = float(roc_auc_score(y_val[val_inc_mask], y_val_prob_genuine[val_inc_mask]))

        results[name] = {
            "artifact_saved": artifact_path,
            "validation_sample_count": len(y_val),
            "genuine_count": int((y_val == 1).sum()),
            "manipulated_count": int((y_val == 0).sum()),
            "default_metrics": {
                "accuracy": round(float(accuracy_score(y_val, y_val_pred_default)), 4),
                "roc_auc": round(float(roc_auc_score(y_val, y_val_prob_genuine)), 4),
                "confusion_matrix": cm_def.tolist(),
                "fpr_on_genuine": round(float(fn / max(1, fn + tp)), 4),
                "manipulated_recall": round(float(tn / max(1, tn + fp)), 4),
                "manipulated_precision": round(float(precision_score(y_val, y_val_pred_default, pos_label=0, zero_division=0)), 4),
                "manipulated_f1": round(float(f1_score(y_val, y_val_pred_default, pos_label=0, zero_division=0)), 4),
                "genuine_f1": round(float(f1_score(y_val, y_val_pred_default, pos_label=1, zero_division=0)), 4)
            },
            "calibration_diagnostics": {
                "brier_score": round(brier, 4),
                "log_loss": round(logloss, 4),
                "reliability_curve_bins": {
                    "mean_predicted_prob_genuine": [round(float(p), 4) for p in prob_pred],
                    "true_fraction_genuine": [round(float(t), 4) for t in prob_true]
                }
            },
            "threshold_sweep_p_manip": sweep_data,
            "end_to_end_quality_breakdown": {
                "reliable_subset": {
                    "count": int(val_rel_mask.sum()),
                    "real": int((y_val[val_rel_mask] == 1).sum()),
                    "fake": int((y_val[val_rel_mask] == 0).sum()),
                    "accuracy": round(float(accuracy_score(y_val[val_rel_mask], y_val_pred_default[val_rel_mask])), 4),
                    "roc_auc": round(auc_rel, 4),
                    "fpr_on_genuine": round(float(fn_r / max(1, fn_r + tp_r)), 4),
                    "manipulated_recall": round(float(tn_r / max(1, tn_r + fp_r)), 4),
                    "confusion_matrix": cm_rel.tolist()
                },
                "inconclusive_subset": {
                    "count": int(val_inc_mask.sum()),
                    "real": int((y_val[val_inc_mask] == 1).sum()),
                    "fake": int((y_val[val_inc_mask] == 0).sum()),
                    "accuracy": round(float(accuracy_score(y_val[val_inc_mask], y_val_pred_default[val_inc_mask])), 4),
                    "roc_auc": round(auc_inc, 4),
                    "fpr_on_genuine": round(float(fn_i / max(1, fn_i + tp_i)), 4),
                    "manipulated_recall": round(float(tn_i / max(1, tn_i + fp_i)), 4),
                    "confusion_matrix": cm_inc.tolist()
                }
            }
        }

    report = {
        "experiment_title": "Validation-Based Model Comparison & Threshold Selection",
        "splits_used": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_split_touched": False,
            "test_unseen_touched": False
        },
        "feature_count": len(feature_cols),
        "results": results
    }

    with open(val_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Validation experiments report saved to {val_report_path}")
    return report

if __name__ == "__main__":
    run_validation_experiments()
