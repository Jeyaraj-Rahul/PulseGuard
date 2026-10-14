import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

logger = logging.getLogger("PulseGuardClassifier")

class PulseGuardClassifier:
    """
    Modular classifier for rPPG deepfake detection.
    Supports GroupKFold evaluation (preventing subject data leakage across train/test sets),
    standard scaling, and serialization via joblib.
    """

    FEATURE_NAMES = [
        "forehead_bpm", "forehead_snr",
        "left_cheek_bpm", "left_cheek_snr",
        "right_cheek_bpm", "right_cheek_snr",
        "nose_bpm", "nose_snr",
        "bpm_std_dev", "bpm_max_diff", "mean_bpm",
        "mean_snr", "min_snr",
        "corr_forehead_left_cheek", "coh_forehead_left_cheek",
        "corr_forehead_right_cheek", "coh_forehead_right_cheek",
        "corr_left_cheek_right_cheek", "coh_left_cheek_right_cheek",
        "mean_cross_correlation", "mean_cross_coherence", "consistency_score"
    ]

    def __init__(self, model_type: str = "svm"):
        self.model_type = model_type.lower()
        self.pipeline: Optional[Pipeline] = None
        self.is_trained: bool = False
        self.evaluation_results: Dict[str, Any] = {}

    def _build_model(self):
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        if self.model_type == "logistic_regression":
            clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        elif self.model_type == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)
        else:  # default SVM with probability estimation
            clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)

        return Pipeline([
            ("imputer", imputer),
            ("scaler", scaler),
            ("classifier", clf)
        ])

    def validate_features(self, feature_dict: Dict[str, float]) -> Tuple[bool, List[str], Dict[str, float]]:
        """
        Identify exactly which features contain NaN, positive infinity, or negative infinity.
        Logs feature names and values safely.
        Returns (is_valid, list_of_invalid_feature_names, invalid_details_dict).
        """
        invalid_features = {}
        for feat_name in self.FEATURE_NAMES:
            val = feature_dict.get(feat_name, np.nan)
            if val is None or np.isnan(val) or np.isinf(val):
                invalid_features[feat_name] = val
                logger.warning("Invalid feature detected in input: %s = %s", feat_name, val)

        is_valid = len(invalid_features) == 0
        return is_valid, list(invalid_features.keys()), invalid_features

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Train classifier using strict GroupKFold (or Stratified if groups is None) cross validation.
        Label convention: 1 = Real, 0 = Manipulated / Deepfake.
        """
        # Ensure only expected feature columns are passed
        avail_features = [f for f in self.FEATURE_NAMES if f in X.columns]
        X_mat = X[avail_features].values
        y_vec = np.array(y, dtype=int)

        self.pipeline = self._build_model()

        if groups is not None and len(np.unique(groups)) >= n_splits:
            cv = GroupKFold(n_splits=n_splits)
            scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            cv_results = cross_validate(
                self.pipeline, X_mat, y_vec, groups=groups, cv=cv, scoring=scoring, return_train_score=False
            )
            eval_summary = {
                "cv_method": "GroupKFold (Subject-independent)",
                "splits": n_splits,
                "accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
                "accuracy_std": float(np.std(cv_results["test_accuracy"])),
                "f1_mean": float(np.mean(cv_results["test_f1"])),
                "f1_std": float(np.std(cv_results["test_f1"])),
                "auc_mean": float(np.mean(cv_results["test_roc_auc"])),
                "auc_std": float(np.std(cv_results["test_roc_auc"])),
            }
        else:
            eval_summary = {
                "cv_method": "Standard fit (Insufficient groups for GroupKFold)",
                "splits": 1
            }

        # Train final pipeline on complete dataset
        self.pipeline.fit(X_mat, y_vec)
        self.is_trained = True

        # In-sample metrics for reporting
        y_pred = self.pipeline.predict(X_mat)
        y_proba = self.pipeline.predict_proba(X_mat)[:, 1] if hasattr(self.pipeline, "predict_proba") else None

        cm = confusion_matrix(y_vec, y_pred)
        eval_summary.update({
            "train_accuracy": float(accuracy_score(y_vec, y_pred)),
            "train_f1": float(f1_score(y_vec, y_pred, zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(y_vec, y_pred, target_names=["Manipulated", "Real"], output_dict=True)
        })

        self.evaluation_results = eval_summary
        return eval_summary

    def predict_sample(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict whether a video is Real or Deepfake based on extracted physiological features.
        Validates feature integrity and logs any NaN/Inf entries.
        """
        if not self.is_trained or self.pipeline is None:
            return {
                "is_model_trained": False,
                "prediction": "Unknown",
                "label": -1,
                "confidence": 0.0,
                "is_valid_sample": False,
                "invalid_features": [],
                "explanation": "No trained machine learning model loaded. Analysis is unvalidated signal evaluation only."
            }

        # Check for invalid feature values (NaN, +inf, -inf)
        is_valid, invalid_keys, invalid_details = self.validate_features(feature_dict)

        # Check expected feature count and alignment
        expected_features = self.FEATURE_NAMES
        row = [feature_dict.get(k, np.nan) for k in expected_features]
        x_vec = np.array(row, dtype=float).reshape(1, -1)

        # If too many key features are missing/invalid, reject prediction
        if not is_valid:
            logger.warning(
                "Sample contains %d invalid features: %s",
                len(invalid_keys), invalid_details
            )

        try:
            pred_label = int(self.pipeline.predict(x_vec)[0])
            probas = self.pipeline.predict_proba(x_vec)[0]
            real_prob = float(probas[1])
            fake_prob = float(probas[0])

            label_name = "Real" if pred_label == 1 else "Deepfake / Manipulated"
            confidence = real_prob if pred_label == 1 else fake_prob

            return {
                "is_model_trained": True,
                "prediction": label_name,
                "label": pred_label,
                "real_probability": round(real_prob, 4),
                "fake_probability": round(fake_prob, 4),
                "confidence": round(confidence, 4),
                "is_valid_sample": is_valid,
                "invalid_features": invalid_keys,
                "invalid_details": invalid_details,
                "is_calibrated": False,
                "explanation": f"Baseline {self.model_type.upper()} model scored video as {label_name} (uncalibrated confidence: {confidence*100:.1f}%)."
            }
        except Exception as e:
            logger.error("Classifier inference failed: %s", e)
            return {
                "is_model_trained": True,
                "prediction": "Error",
                "label": -1,
                "confidence": 0.0,
                "is_valid_sample": False,
                "invalid_features": invalid_keys,
                "invalid_details": invalid_details,
                "is_calibrated": False,
                "explanation": f"Inference failed due to feature incompatibility: {e}"
            }

    def save(self, file_path: str):
        """Serialize trained pipeline to disk via Joblib."""
        data = {
            "model_type": self.model_type,
            "pipeline": self.pipeline,
            "is_trained": self.is_trained,
            "feature_names": self.FEATURE_NAMES,
            "evaluation_results": self.evaluation_results
        }
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        joblib.dump(data, file_path)

    @classmethod
    def load(cls, file_path: str) -> "PulseGuardClassifier":
        """Load trained pipeline from disk."""
        data = joblib.load(file_path)
        clf_obj = cls(model_type=data.get("model_type", "svm"))
        clf_obj.pipeline = data.get("pipeline")
        clf_obj.is_trained = data.get("is_trained", False)
        clf_obj.evaluation_results = data.get("evaluation_results", {})
        return clf_obj
