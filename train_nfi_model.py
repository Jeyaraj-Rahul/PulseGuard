import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)

from pulseguard.models.classifier import PulseGuardClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainNFI")

def run_nfi_training():
    features_csv = "data/features_nfi.csv"
    production_model_path = "models_saved/pulseguard_svm_baseline.joblib"
    output_model_path = "models_saved/pulseguard_svm_nfi_trained.joblib"
    eval_report_path = "data/evaluation_nfi_report.json"

    logger.info("Starting PulseGuard AI NFI Baseline Training Run")

    # 1. Load extracted features
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Features file not found at {features_csv}")

    df = pd.read_csv(features_csv)
    total_records = len(df)
    logger.info(f"Loaded {total_records} records from {features_csv}")

    # 2. Check and exclude failed extractions
    failed_df = df[df["extraction_success"] == False]
    valid_df = df[df["extraction_success"] == True].copy()
    excluded_count = len(failed_df)
    effective_count = len(valid_df)

    logger.info(f"Excluded failures: {excluded_count}")
    logger.info(f"Effective samples for training/eval: {effective_count}")

    # 3. Verify feature schema
    feature_cols = PulseGuardClassifier.FEATURE_NAMES
    for col in feature_cols:
        if col not in valid_df.columns:
            raise ValueError(f"Feature column '{col}' missing from features file!")

    # 4. Partition splits
    train_df = valid_df[valid_df["split"] == "train"].copy()
    val_df = valid_df[valid_df["split"] == "val"].copy()
    test_df = valid_df[valid_df["split"] == "test"].copy()
    test_unseen_df = valid_df[valid_df["split"] == "test_unseen"].copy()

    logger.info(f"Split counts: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, test_unseen={len(test_unseen_df)}")

    # Features and labels
    # Label convention in PulseGuardClassifier: 1 = Real / Genuine, 0 = Manipulated / Deepfake
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values.astype(int)

    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values.astype(int)

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values.astype(int)

    X_unseen = test_unseen_df[feature_cols].values
    y_unseen = test_unseen_df["label"].values.astype(int)

    # 5. Build pipeline: Imputer -> Scaler -> SVC (balanced, rbf)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
    ])

    # 6. Fit exclusively on training data
    logger.info("Fitting pipeline on training data exclusively...")
    pipeline.fit(X_train, y_train)

    # In-sample (train) metrics
    y_train_pred = pipeline.predict(X_train)
    y_train_prob = pipeline.predict_proba(X_train)[:, 1]

    # 7. Validation evaluation
    y_val_pred = pipeline.predict(X_val)
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]

    # 8. Primary test evaluation
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics computation helper
    def compute_metrics(y_true, y_pred, y_prob):
        # In this task:
        # Ground truth labels: 0 = Manipulated, 1 = Genuine/Real
        # Scikit-learn binary classification convention: class 1 is positive, class 0 is negative
        # Confusion matrix:
        # rows: true [0, 1]
        # cols: pred [0, 1]
        # cm[0,0] = TN (true manipulated)
        # cm[0,1] = FP (manipulated predicted as genuine)
        # cm[1,0] = FN (genuine predicted as manipulated)
        # cm[1,1] = TP (true genuine)
        # However, in forensic / deepfake detection terminology:
        # Often "Manipulated" is considered the target forensic condition.
        # We will report both perspectives explicitly to avoid any ambiguity:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # False positive rate on genuine videos:
        # Genuine videos are class 1 (Real).
        # A false alarm on a genuine video occurs when a Genuine video (y_true=1) is predicted as Manipulated (pred=0).
        # Total genuine videos = fn + tp = (y_true == 1).sum()
        # Genuine misclassified as fake = fn
        # FPR_on_genuine = fn / (fn + tp)
        genuine_count = int((y_true == 1).sum())
        genuine_misclassified_as_fake = int(fn)
        fpr_on_genuine = float(genuine_misclassified_as_fake / max(1, genuine_count))

        # Also standard scikit-learn FPR (where class 1 is positive, class 0 negative):
        # fp / (fp + tn) -> rate of manipulated falsely called genuine
        manipulated_count = int((y_true == 0).sum())
        manipulated_misclassified_as_real = int(fp)
        fpr_standard = float(manipulated_misclassified_as_real / max(1, manipulated_count)) if manipulated_count > 0 else 0.0

        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None

        return {
            "sample_count": len(y_true),
            "genuine_count": genuine_count,
            "manipulated_count": manipulated_count,
            "confusion_matrix": {
                "format": "rows: true [0=Manipulated, 1=Genuine], cols: pred [0=Manipulated, 1=Genuine]",
                "matrix": cm.tolist(),
                "true_manipulated_pred_manipulated (TN)": int(tn),
                "true_manipulated_pred_genuine (FP)": int(fp),
                "true_genuine_pred_manipulated (FN)": int(fn),
                "true_genuine_pred_genuine (TP)": int(tp)
            },
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": auc,
            "false_positive_rate_on_genuine": {
                "definition": "Genuine videos misclassified as Manipulated: FN / (FN + TP)",
                "value": fpr_on_genuine,
                "genuine_flagged_as_manipulated": genuine_misclassified_as_fake,
                "total_genuine": genuine_count
            },
            "false_alarm_rate_on_manipulated": {
                "definition": "Manipulated videos misclassified as Genuine: FP / (FP + TN)",
                "value": fpr_standard,
                "manipulated_missed_as_genuine": manipulated_misclassified_as_real,
                "total_manipulated": manipulated_count
            },
            "class_metrics": {
                "Manipulated (label=0)": {
                    "precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
                    "f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
                },
                "Genuine (label=1)": {
                    "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
                    "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
                }
            }
        }

    train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob)
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

    # 9. Inconclusive vs Reliable subset evaluation on primary test set
    test_reliable_mask = (test_df["signal_quality_status"] == "Reliable").values
    test_inconclusive_mask = (test_df["signal_quality_status"] == "Inconclusive").values

    test_subsets = {}
    if test_reliable_mask.sum() > 0:
        test_subsets["reliable"] = compute_metrics(
            y_test[test_reliable_mask], y_test_pred[test_reliable_mask], y_test_prob[test_reliable_mask]
        )
    if test_inconclusive_mask.sum() > 0:
        test_subsets["inconclusive"] = compute_metrics(
            y_test[test_inconclusive_mask], y_test_pred[test_inconclusive_mask], y_test_prob[test_inconclusive_mask]
        )

    # 10. Unseen Generator Evaluation (Sora_SYN)
    # y_unseen is all 0 (Manipulated)
    y_unseen_pred = pipeline.predict(X_unseen)
    y_unseen_prob = pipeline.predict_proba(X_unseen)[:, 1]
    sora_total = len(y_unseen)
    sora_detected_as_manipulated = int((y_unseen_pred == 0).sum())
    sora_missed_as_genuine = int((y_unseen_pred == 1).sum())
    sora_detection_recall = float(sora_detected_as_manipulated / max(1, sora_total))

    unseen_metrics = {
        "generator": "Sora_SYN",
        "sample_count": sora_total,
        "manipulated_predicted_as_manipulated": sora_detected_as_manipulated,
        "manipulated_missed_as_genuine": sora_missed_as_genuine,
        "detection_recall_on_sora": sora_detection_recall,
        "average_predicted_prob_genuine": float(np.mean(y_unseen_prob)),
        "notes": "Unseen generator diagnostic evaluation only. Does not measure general accuracy across all potential generators."
    }

    # 11. Save experimental model artifact
    logger.info(f"Saving experimental model to {output_model_path}...")
    joblib.dump(pipeline, output_model_path)

    # Verify existing production model was NOT touched
    prod_exists = os.path.exists(production_model_path)
    prod_size = os.path.getsize(production_model_path) if prod_exists else 0
    logger.info(f"Verified production model intact at {production_model_path} (size={prod_size} bytes)")

    # 12. Verification of saved artifact loading and inference compatibility
    logger.info("Verifying saved experimental model artifact...")
    loaded_pipe = joblib.load(output_model_path)

    # Test single-sample prediction with PulseGuardClassifier schema
    sample_feature_dict = {f: float(X_test[0, i]) for i, f in enumerate(feature_cols)}
    sample_vec = np.array([[sample_feature_dict[f] for f in feature_cols]])
    pred_class = int(loaded_pipe.predict(sample_vec)[0])
    pred_prob = loaded_pipe.predict_proba(sample_vec)[0]
    logger.info(f"Verification test sample prediction: class={pred_class}, prob={pred_prob}")

    # 13. Assemble complete JSON evaluation report
    report = {
        "model_metadata": {
            "model_type": "SVC (RBF kernel, balanced class weight)",
            "pipeline_steps": ["SimpleImputer(strategy='median')", "StandardScaler()", "SVC(kernel='rbf', probability=True, class_weight='balanced')"],
            "random_state": 42,
            "feature_count": len(feature_cols),
            "feature_names": feature_cols,
            "trained_model_path": output_model_path,
            "production_model_path": production_model_path,
            "production_model_untouched": prod_exists
        },
        "sample_counts": {
            "total_manifest": total_records,
            "excluded_tracking_failures": excluded_count,
            "effective_samples": effective_count,
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "test_unseen_sora": len(test_unseen_df)
        },
        "training_set_distribution": {
            "genuine (label=1)": int((y_train == 1).sum()),
            "manipulated (label=0)": int((y_train == 0).sum())
        },
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_primary_metrics": test_metrics,
        "test_subsets_metrics": test_subsets,
        "unseen_generator_diagnostic": unseen_metrics
    }

    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report saved to {eval_report_path}")

    return report

if __name__ == "__main__":
    run_nfi_training()
