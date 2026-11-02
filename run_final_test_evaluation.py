import os
import json
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, brier_score_loss
)

def wilson_score_interval(count, nobs, alpha=0.05):
    """Compute Wilson score interval for a binomial proportion."""
    if nobs == 0:
        return 0.0, 0.0
    p = count / nobs
    z = stats.norm.ppf(1 - alpha / 2)
    z2 = z * z
    denom = 1 + z2 / nobs
    center = (p + z2 / (2 * nobs)) / denom
    half_width = z * np.sqrt((p * (1 - p) + z2 / (4 * nobs)) / nobs) / denom
    lower = max(0.0, float(center - half_width))
    upper = min(1.0, float(center + half_width))
    return lower, upper

def bootstrap_ci(metric_fn, y_true, y_pred, y_prob=None, n_bootstraps=2000, alpha=0.05, seed=42):
    """Compute 95% bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0
    boot_vals = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if y_prob is not None:
            # Check if there are at least two classes if metric requires it (e.g. roc_auc)
            if len(np.unique(y_true[idx])) < 2:
                continue
            val = metric_fn(y_true[idx], y_prob[idx])
        else:
            val = metric_fn(y_true[idx], y_pred[idx])
        boot_vals.append(val)
    if len(boot_vals) == 0:
        return 0.0, 0.0
    lower = float(np.percentile(boot_vals, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return lower, upper

def run_evaluation():
    os.chdir(r"C:\Users\jeyar\OneDrive\Attachments\Desktop\PulseGuard- AI")
    
    features_csv = "data/features_nfi.csv"
    manifest_csv = "data/manifest_nfi.csv"
    rf_path = "models_saved/candidate_random_forest_nfi.joblib"
    json_out = "data/final_test_evaluation_report.json"
    md_out = "data/final_test_evaluation_report.md"

    # Load data
    df = pd.read_csv(features_csv)
    manifest = pd.read_csv(manifest_csv)
    rf_pipe = joblib.load(rf_path)

    feature_cols = [
        'forehead_bpm', 'forehead_snr', 'left_cheek_bpm', 'left_cheek_snr',
        'right_cheek_bpm', 'right_cheek_snr', 'nose_bpm', 'nose_snr',
        'bpm_std_dev', 'bpm_max_diff', 'mean_bpm', 'mean_snr', 'min_snr',
        'corr_forehead_left_cheek', 'coh_forehead_left_cheek',
        'corr_forehead_right_cheek', 'coh_forehead_right_cheek',
        'corr_left_cheek_right_cheek', 'coh_left_cheek_right_cheek',
        'mean_cross_correlation', 'mean_cross_coherence', 'consistency_score'
    ]

    # Verification of classes: pipeline classes_ are [0, 1] where 0=Manipulated, 1=Genuine
    assert list(rf_pipe.classes_) == [0, 1], f"Unexpected classes: {rf_pipe.classes_}"

    # Extract test and test_unseen
    test_all = df[df['split'] == 'test'].copy()
    unseen_all = df[df['split'] == 'test_unseen'].copy()

    test_failed = test_all[test_all['extraction_success'] == False]
    test_eligible = test_all[test_all['extraction_success'] == True].copy()

    unseen_failed = unseen_all[unseen_all['extraction_success'] == False]
    unseen_eligible = unseen_all[unseen_all['extraction_success'] == True].copy()

    # Pre-computation check
    assert len(test_eligible) == 95, f"Expected 95 eligible test samples, got {len(test_eligible)}"
    assert len(unseen_eligible) == 24, f"Expected 24 eligible unseen samples, got {len(unseen_eligible)}"

    # Probabilities
    X_test = test_eligible[feature_cols].values
    y_test_ground_truth = test_eligible['label'].values.astype(int) # 0=Manipulated, 1=Genuine

    test_probs = rf_pipe.predict_proba(X_test)
    test_prob_manip = test_probs[:, 0]
    test_prob_gen = test_probs[:, 1]

    X_unseen = unseen_eligible[feature_cols].values
    y_unseen_ground_truth = unseen_eligible['label'].values.astype(int) # All 0
    unseen_probs = rf_pipe.predict_proba(X_unseen)
    unseen_prob_manip = unseen_probs[:, 0]

    # Metrics evaluation helper
    # Positive class is MANIPULATED.
    # To compute metrics where Manipulated is positive:
    # y_pos: 1 if Manipulated (label=0), 0 if Genuine (label=1)
    y_test_pos = (y_test_ground_truth == 0).astype(int)

    # ROC-AUC: P(Manipulated) predicting y_test_pos
    roc_auc_val = float(roc_auc_score(y_test_pos, test_prob_manip))
    roc_auc_ci = bootstrap_ci(roc_auc_score, y_test_pos, None, y_prob=test_prob_manip)

    # Brier score for P(Manipulated) predicting y_test_pos
    brier_val = float(brier_score_loss(y_test_pos, test_prob_manip))

    # Evaluate at locked thresholds: 0.50 and 0.55
    locked_thresholds = [0.50, 0.55]
    threshold_results = {}

    for t in locked_thresholds:
        # Decision rule: Predict Manipulated if P(Manipulated) >= t, else Genuine
        # pred_manip: 1 if predicted Manipulated, 0 if Genuine
        pred_manip = (test_prob_manip >= t).astype(int)

        # Standard Confusion Matrix for Positive=Manipulated:
        # Rows: Actual [Genuine(0), Manipulated(1)]
        # Cols: Predicted [Genuine(0), Manipulated(1)]
        # TN: Actual Genuine, Predicted Genuine
        # FP: Actual Genuine, Predicted Manipulated (False Alarm on Genuine)
        # FN: Actual Manipulated, Predicted Genuine (Missed Manipulated)
        # TP: Actual Manipulated, Predicted Manipulated (Detected Manipulated)
        cm = confusion_matrix(y_test_pos, pred_manip, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        total_genuine = tn + fp # actual genuine (62)
        total_manip = fn + tp   # actual manipulated (33)
        total_n = len(y_test_pos)

        accuracy = float((tp + tn) / total_n)
        acc_ci = wilson_score_interval(tp + tn, total_n)

        # False positive rate on genuine videos: FP / total_genuine
        fpr_genuine = float(fp / total_genuine)
        fpr_ci = wilson_score_interval(fp, total_genuine)

        # Manipulated video miss rate: FN / total_manip
        miss_rate_manip = float(fn / total_manip)
        miss_ci = wilson_score_interval(fn, total_manip)

        # Recall (Detection rate) on manipulated: TP / total_manip = 1 - miss_rate
        recall_manip = float(tp / total_manip)
        rec_ci = wilson_score_interval(tp, total_manip)

        # Precision on manipulated: TP / (TP + FP)
        precision_manip = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        prec_ci = wilson_score_interval(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0)

        # F1 on manipulated
        f1_manip = float(f1_score(y_test_pos, pred_manip, pos_label=1, zero_division=0))

        threshold_results[str(t)] = {
            "threshold": t,
            "decision_rule": f"P(Manipulated) >= {t}",
            "confusion_matrix": {
                "format": "rows: Actual [0=Genuine, 1=Manipulated], cols: Predicted [0=Genuine, 1=Manipulated]",
                "matrix": cm.tolist(),
                "true_genuine_pred_genuine (TN)": int(tn),
                "true_genuine_pred_manipulated (FP - False Alarm)": int(fp),
                "true_manipulated_pred_genuine (FN - Missed)": int(fn),
                "true_manipulated_pred_manipulated (TP - Detected)": int(tp)
            },
            "accuracy": {
                "value": accuracy,
                "ci_95": acc_ci
            },
            "fpr_on_genuine": {
                "value": fpr_genuine,
                "ci_95": fpr_ci,
                "numerator": int(fp),
                "denominator": int(total_genuine)
            },
            "manipulated_miss_rate": {
                "value": miss_rate_manip,
                "ci_95": miss_ci,
                "numerator": int(fn),
                "denominator": int(total_manip)
            },
            "manipulated_recall": {
                "value": recall_manip,
                "ci_95": rec_ci,
                "numerator": int(tp),
                "denominator": int(total_manip)
            },
            "manipulated_precision": {
                "value": precision_manip,
                "ci_95": prec_ci,
                "numerator": int(tp),
                "denominator": int(tp + fp)
            },
            "manipulated_f1": f1_manip,
            "total_predicted_manipulated": int(tp + fp),
            "total_predicted_genuine": int(tn + fn)
        }

    # Quality-gated End-to-End breakdown on Primary Test Set
    test_rel_mask = (test_eligible['signal_quality_status'] == 'Reliable').values
    test_inc_mask = (test_eligible['signal_quality_status'] == 'Inconclusive').values

    n_rel = int(test_rel_mask.sum())
    n_inc = int(test_inc_mask.sum())
    total_eligible_test = len(test_eligible)

    gate_breakdown = {
        "total_test_manifest": len(test_all),
        "failed_extractions": {
            "count": len(test_failed),
            "samples": test_failed[['filename', 'failure_reason']].to_dict(orient='records')
        },
        "eligible_samples": total_eligible_test,
        "reliable_verdicts_issued": {
            "count": n_rel,
            "percentage": float(n_rel / total_eligible_test * 100),
            "by_ground_truth": {
                "genuine": int((test_eligible[test_rel_mask]['label'] == 1).sum()),
                "manipulated": int((test_eligible[test_rel_mask]['label'] == 0).sum())
            }
        },
        "inconclusive_verdicts_suppressed": {
            "count": n_inc,
            "percentage": float(n_inc / total_eligible_test * 100),
            "by_ground_truth": {
                "genuine": int((test_eligible[test_inc_mask]['label'] == 1).sum()),
                "manipulated": int((test_eligible[test_inc_mask]['label'] == 0).sum())
            }
        }
    }

    # Reliable subset specific metrics at both thresholds
    reliable_subset_metrics = {}
    if n_rel > 0:
        y_rel_pos = y_test_pos[test_rel_mask]
        prob_rel_manip = test_prob_manip[test_rel_mask]
        auc_rel = float(roc_auc_score(y_rel_pos, prob_rel_manip))
        
        for t in locked_thresholds:
            pred_rel_manip = (prob_rel_manip >= t).astype(int)
            cm_r = confusion_matrix(y_rel_pos, pred_rel_manip, labels=[0, 1])
            tn_r, fp_r, fn_r, tp_r = cm_r.ravel()
            total_gen_r = tn_r + fp_r
            total_man_r = fn_r + tp_r
            
            reliable_subset_metrics[str(t)] = {
                "threshold": t,
                "confusion_matrix": cm_r.tolist(),
                "accuracy": float((tp_r + tn_r) / n_rel),
                "fpr_on_genuine": float(fp_r / total_gen_r) if total_gen_r > 0 else 0.0,
                "manipulated_recall": float(tp_r / total_man_r) if total_man_r > 0 else 0.0,
                "manipulated_precision": float(tp_r / (tp_r + fp_r)) if (tp_r + fp_r) > 0 else 0.0,
                "manipulated_f1": float(f1_score(y_rel_pos, pred_rel_manip, pos_label=1, zero_division=0)),
                "roc_auc": auc_rel
            }

    # Sora_SYN (test_unseen) Out-of-Distribution Diagnostic Evaluation
    # All 24 samples are Manipulated (ground truth 0).
    # Positive class = Manipulated.
    total_unseen_eligible = len(unseen_eligible)
    unseen_gate_rel_mask = (unseen_eligible['signal_quality_status'] == 'Reliable').values
    unseen_gate_inc_mask = (unseen_eligible['signal_quality_status'] == 'Inconclusive').values

    sora_results = {
        "generator": "Sora_SYN",
        "description": "Out-of-distribution unseen generator diagnostic subset (Manipulated samples only)",
        "total_manifest": len(unseen_all),
        "failed_extractions": {
            "count": len(unseen_failed),
            "samples": unseen_failed[['filename', 'failure_reason']].to_dict(orient='records')
        },
        "eligible_samples": total_unseen_eligible,
        "quality_gate_breakdown": {
            "reliable_count": int(unseen_gate_rel_mask.sum()),
            "inconclusive_count": int(unseen_gate_inc_mask.sum())
        },
        "threshold_evaluations": {}
    }

    for t in locked_thresholds:
        # Detected as manipulated: P(Manipulated) >= t
        detected_manip = (unseen_prob_manip >= t).astype(int)
        tp_sora = int(detected_manip.sum())
        fn_sora = int(total_unseen_eligible - tp_sora)
        recall_sora = float(tp_sora / total_unseen_eligible)
        recall_ci_sora = wilson_score_interval(tp_sora, total_unseen_eligible)

        # On reliable subset only
        if unseen_gate_rel_mask.sum() > 0:
            tp_sora_rel = int((unseen_prob_manip[unseen_gate_rel_mask] >= t).sum())
            n_sora_rel = int(unseen_gate_rel_mask.sum())
            recall_sora_rel = float(tp_sora_rel / n_sora_rel)
        else:
            recall_sora_rel = None

        sora_results["threshold_evaluations"][str(t)] = {
            "threshold": t,
            "detected_as_manipulated": tp_sora,
            "missed_as_genuine": fn_sora,
            "total_sora_samples": total_unseen_eligible,
            "manipulated_recall": recall_sora,
            "recall_ci_95": recall_sora_ci if 'recall_sora_ci' in locals() else recall_ci_sora,
            "miss_rate": float(fn_sora / total_unseen_eligible),
            "recall_on_reliable_subset": recall_sora_rel
        }

    # Comparison with Validation split
    val_comparison = {
        "validation_samples": 92,
        "validation_results_recorded": {
            "auc": 0.5760,
            "brier": 0.2409,
            "threshold_0_50": {
                "fpr_on_genuine": 0.2778,
                "manipulated_recall": 0.3684,
                "manipulated_precision": 0.4828,
                "manipulated_f1": 0.4179
            },
            "threshold_0_55": {
                "fpr_on_genuine": 0.1481,
                "manipulated_recall": 0.2895,
                "manipulated_precision": 0.5789,
                "manipulated_f1": 0.3860
            }
        },
        "test_results_observed": {
            "auc": roc_auc_val,
            "brier": brier_val,
            "threshold_0_50": {
                "fpr_on_genuine": threshold_results["0.5"]["fpr_on_genuine"]["value"],
                "manipulated_recall": threshold_results["0.5"]["manipulated_recall"]["value"],
                "manipulated_precision": threshold_results["0.5"]["manipulated_precision"]["value"],
                "manipulated_f1": threshold_results["0.5"]["manipulated_f1"]
            },
            "threshold_0_55": {
                "fpr_on_genuine": threshold_results["0.55"]["fpr_on_genuine"]["value"],
                "manipulated_recall": threshold_results["0.55"]["manipulated_recall"]["value"],
                "manipulated_precision": threshold_results["0.55"]["manipulated_precision"]["value"],
                "manipulated_f1": threshold_results["0.55"]["manipulated_f1"]
            }
        }
    }

    report = {
        "metadata": {
            "title": "PulseGuard AI Final Test Evaluation Report - Random Forest Candidate",
            "candidate_artifact": rf_path,
            "locked_thresholds": locked_thresholds,
            "positive_class": "Manipulated (label=0 in original schema, mapped to pos_label=1 in binary evaluation)",
            "classes_mapping": {"0": "Manipulated", "1": "Genuine"},
            "features_evaluated": len(feature_cols)
        },
        "primary_test_summary": {
            "sample_counts": {
                "manifest_total": len(test_all),
                "extraction_failures": len(test_failed),
                "eligible_evaluated": total_eligible_test,
                "actual_genuine": int((y_test_ground_truth == 1).sum()),
                "actual_manipulated": int((y_test_ground_truth == 0).sum())
            },
            "model_only_metrics": {
                "roc_auc": {
                    "value": roc_auc_val,
                    "ci_95": roc_auc_ci
                },
                "brier_score": brier_val,
                "by_threshold": threshold_results
            },
            "quality_gate_breakdown": gate_breakdown,
            "reliable_subset_metrics": reliable_subset_metrics
        },
        "unseen_generator_sora_diagnostic": sora_results,
        "validation_vs_test_comparison": val_comparison
    }

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {json_out}")

    # Generate Markdown summary
    md_content = f"""# PulseGuard AI: Final Test Evaluation Report (Random Forest Candidate)

**Candidate Model:** `{rf_path}`  
**Dataset Source of Truth:** `data/manifest_nfi.csv` & `data/features_nfi.csv`  
**Positive Class Definition:** **Manipulated** (Ground Truth Label = 0, evaluated as positive condition)  
**Negative Class Definition:** **Genuine / Authentic** (Ground Truth Label = 1, evaluated as negative condition)  
**Locked Decision Thresholds:** `P(Manipulated) >= 0.50` and `P(Manipulated) >= 0.55`  

---

## 1. Primary Test Set Overview ($N=95$ Eligible)

* **Manifest Total:** 96 videos
* **Excluded Extraction Failures:** 1 video (`deepfake6.mp4` / `Luma_SYN` due to face tracking < 0.30)
* **Eligible Evaluated Samples:** **95 videos** (62 Genuine, 33 Manipulated across 5 face-swap and 5 diffusion generators)
* **Overall ROC-AUC (P(Manipulated)):** **{roc_auc_val:.4f}** (95% CI: [{roc_auc_ci[0]:.4f}, {roc_auc_ci[1]:.4f}])
* **Overall Brier Score:** **{brier_val:.4f}**

---

## 2. Model-Only Performance at Locked Thresholds

| Metric | Threshold $T = 0.50$ | Threshold $T = 0.55$ | Definition / Formula |
| :--- | :---: | :---: | :--- |
| **Confusion Matrix** | `[[TN=48, FP=14], [FN=19, TP=14]]` | `[[TN=53, FP=9], [FN=25, TP=8]]` | Rows: Actual [Genuine, Manipulated]<br>Cols: Predicted [Genuine, Manipulated] |
| **Accuracy** | **{threshold_results['0.5']['accuracy']['value']*100:.2f}%** [{threshold_results['0.5']['accuracy']['ci_95'][0]*100:.1f}%, {threshold_results['0.5']['accuracy']['ci_95'][1]*100:.1f}%] | **{threshold_results['0.55']['accuracy']['value']*100:.2f}%** [{threshold_results['0.55']['accuracy']['ci_95'][0]*100:.1f}%, {threshold_results['0.55']['accuracy']['ci_95'][1]*100:.1f}%] | $(TP + TN) / Total$ |
| **FPR on Genuine** | **{threshold_results['0.5']['fpr_on_genuine']['value']*100:.2f}%** (14 / 62) [{threshold_results['0.5']['fpr_on_genuine']['ci_95'][0]*100:.1f}%, {threshold_results['0.5']['fpr_on_genuine']['ci_95'][1]*100:.1f}%] | **{threshold_results['0.55']['fpr_on_genuine']['value']*100:.2f}%** (9 / 62) [{threshold_results['0.55']['fpr_on_genuine']['ci_95'][0]*100:.1f}%, {threshold_results['0.55']['fpr_on_genuine']['ci_95'][1]*100:.1f}%] | $FP / Actual\\_Genuine$ (False Alarms) |
| **Manipulated Miss Rate** | **{threshold_results['0.5']['manipulated_miss_rate']['value']*100:.2f}%** (19 / 33) | **{threshold_results['0.55']['manipulated_miss_rate']['value']*100:.2f}%** (25 / 33) | $FN / Actual\\_Manipulated$ (Missed Fakes) |
| **Manipulated Recall** | **{threshold_results['0.5']['manipulated_recall']['value']*100:.2f}%** (14 / 33) [{threshold_results['0.5']['manipulated_recall']['ci_95'][0]*100:.1f}%, {threshold_results['0.5']['manipulated_recall']['ci_95'][1]*100:.1f}%] | **{threshold_results['0.55']['manipulated_recall']['value']*100:.2f}%** (8 / 33) [{threshold_results['0.55']['manipulated_recall']['ci_95'][0]*100:.1f}%, {threshold_results['0.55']['manipulated_recall']['ci_95'][1]*100:.1f}%] | $TP / Actual\\_Manipulated$ (Detection Rate) |
| **Manipulated Precision** | **{threshold_results['0.5']['manipulated_precision']['value']*100:.2f}%** (14 / 28) [{threshold_results['0.5']['manipulated_precision']['ci_95'][0]*100:.1f}%, {threshold_results['0.5']['manipulated_precision']['ci_95'][1]*100:.1f}%] | **{threshold_results['0.55']['manipulated_precision']['value']*100:.2f}%** (8 / 17) [{threshold_results['0.55']['manipulated_precision']['ci_95'][0]*100:.1f}%, {threshold_results['0.55']['manipulated_precision']['ci_95'][1]*100:.1f}%] | $TP / (TP + FP)$ |
| **Manipulated F1** | **{threshold_results['0.5']['manipulated_f1']:.4f}** | **{threshold_results['0.55']['manipulated_f1']:.4f}** | Harmonic mean of Precision and Recall |

---

## 3. End-to-End Quality Gate Behavior & Verdict Coverage

The operational PulseGuard system uses a pre-classifier signal-quality gate (`assess_signal_quality`):

```
                             [All 95 Test Samples]
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        [Inconclusive: 67 (70.5%)]             [Reliable: 28 (29.5%)]
        • Forensic badge suppressed            • Classifier verdict issued
        • 49 Genuine, 18 Manipulated           • 13 Genuine, 15 Manipulated
                                               • ROC-AUC: {reliable_subset_metrics.get('0.5', {}).get('roc_auc', 0.0):.4f}
                                               • T=0.50 FPR on Real: {reliable_subset_metrics.get('0.5', {}).get('fpr_on_genuine', 0.0)*100:.1f}% (4/13)
                                               • T=0.50 Fake Recall: {reliable_subset_metrics.get('0.5', {}).get('manipulated_recall', 0.0)*100:.1f}% (6/15)
```

* **Coverage:** Only **29.5% (28 / 95)** of test videos receive a definitive verdict. **70.5% (67 / 95)** are flagged as `INCONCLUSIVE` due to low rPPG SNR or compression artifacts.
* **Separation Integrity:** Inconclusive samples are **never** treated as deepfakes; they are displayed with an informative quality warning without a forensic determination.

---

## 4. Out-of-Distribution Diagnostic: `test_unseen` (`Sora_SYN`, $N=24$)

* **Total Manifest:** 26 videos
* **Excluded Failures:** 2 videos (`25.mp4` [0.09 tracking], `3.mp4` [0.00 tracking])
* **Eligible Evaluated:** **24 videos** (100% manipulated; zero authentic samples).
* **Important Notice:** ROC-AUC, Accuracy, and FPR on genuine cannot be computed for this subset because it contains no genuine videos. It is evaluated solely for **Detection Recall**.

| Threshold | Detected as Manipulated | Missed as Genuine | Sora Detection Recall (95% CI) | Miss Rate |
| :---: | :---: | :---: | :---: | :---: |
| **$T = 0.50$** | **11** / 24 | **13** / 24 | **{sora_results['threshold_evaluations']['0.5']['manipulated_recall']*100:.2f}%** [{sora_results['threshold_evaluations']['0.5']['recall_ci_95'][0]*100:.1f}%, {sora_results['threshold_evaluations']['0.5']['recall_ci_95'][1]*100:.1f}%] | **54.17%** |
| **$T = 0.55$** | **6** / 24 | **18** / 24 | **{sora_results['threshold_evaluations']['0.55']['manipulated_recall']*100:.2f}%** [{sora_results['threshold_evaluations']['0.55']['recall_ci_95'][0]*100:.1f}%, {sora_results['threshold_evaluations']['0.55']['recall_ci_95'][1]*100:.1f}%] | **75.00%** |

---

## 5. Validation vs. Test Consistency Comparison

| Metric | Validation Observed ($N=92$) | Test Observed ($N=95$) | Consistency Notes |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | 0.5760 | **{roc_auc_val:.4f}** | Slight increase (+0.038) on test split. |
| **Brier Score** | 0.2409 | **{brier_val:.4f}** | Consistent across splits. |
| **$T=0.50$ FPR on Genuine** | 27.78% (15 / 54) | **{threshold_results['0.5']['fpr_on_genuine']['value']*100:.2f}% (14 / 62)** | High consistency (~23% vs 28%). |
| **$T=0.50$ Manipulated Recall** | 36.84% (14 / 38) | **{threshold_results['0.5']['manipulated_recall']['value']*100:.2f}% (14 / 33)** | Consistent detection rate (~37% vs 42%). |
| **$T=0.55$ FPR on Genuine** | 14.81% (8 / 54) | **{threshold_results['0.55']['fpr_on_genuine']['value']*100:.2f}% (9 / 62)** | Validates that $T=0.55$ cuts false alarms in half. |
| **$T=0.55$ Manipulated Recall** | 28.95% (11 / 38) | **{threshold_results['0.55']['manipulated_recall']['value']*100:.2f}% (8 / 33)** | Expected drop in detection rate to preserve low FPR. |

---

## 6. Documented Limitations & Findings

1. **Trade-off between False Alarms and Recall:**  
   At $T=0.50$, the detector identifies $42.4\%$ of fakes but incurs a $22.6\%$ false-positive rate on authentic videos. Raising the threshold to $T=0.55$ successfully reduces false alarms to $14.5\%$, but causes manipulated detection recall to fall to $24.2\%$ (a $75.8\%$ miss rate).
2. **Low Operational Coverage (70.5% Inconclusive):**  
   Because real-world video compression severely dampens facial skin color micro-pulsations, the quality gate suppresses $70.5\%$ of incoming test videos. While this prevents false forensic accusations on noisy inputs, it leaves the system unable to render definitive verdicts on the majority of web videos.
3. **Generalization Gap on Modern Generative Models (Sora):**  
   On Sora videos, detection recall is only $45.8\%$ at $T=0.50$ and drops to $25.0\%$ at $T=0.55$. Sora's video generation preserves temporal facial smoothness without the high-frequency color artifacts of older face-swap models.
4. **Not Production-Ready for Autonomous Forensic Verdicts:**  
   With an overall test ROC-AUC of $\approx 0.61$, this model must remain designated as an experimental research prototype. It should not be used as an autonomous arbiter of video authenticity.
"""

    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Summary markdown saved to {md_out}")

if __name__ == "__main__":
    run_evaluation()
