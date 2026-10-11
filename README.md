# PulseGuard AI: Biophysiological rPPG Deepfake Forensic Engine

PulseGuard AI is an intermediate-level research project for Remote Photoplethysmography (rPPG)-based deepfake detection, inspired by principles from Intel & Binghamton University's *FakeCatcher* research (Ciftci et al.).

The system isolates subtle, periodic color fluctuations in facial skin caused by cardiac blood volume pulses, examines multi-region physiological consistency across facial patches (forehead, cheeks, nose), and classifies videos using subject-independent machine learning models.

---

## Architecture Overview

```
                      [ Uploaded Video ]
                              │
                    ┌─────────▼─────────┐
                    │   VideoValidator  │ (Checks duration, FPS, blur, lighting)
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ FaceROIExtractor  │ (MediaPipe 478-point landmark tracking)
                    └─────────┬─────────┘ (Forehead, Left Cheek, Right Cheek, Nose)
                              │
                    ┌─────────▼─────────┐
                    │  SignalProcessor  │ (Detrending, Butterworth Bandpass 0.7-3.0 Hz,
                    └─────────┬─────────┘  Green / POS / CHROM algorithms, FFT spectra)
                              │
                    ┌─────────▼──────────┐
                    │ Feature Extraction │ (Cross-patch coherence, correlation,
                    └─────────┬──────────┘  BPM standard deviation, SNR, consistency)
                              │
                    ┌─────────▼──────────┐
                    │ PulseGuardClassifier│ (GroupKFold Subject-Independent SVM)
                    └────────────────────┘
```

---

## Key Features

1. **Streamlit Interactive Forensic Dashboard**:
   - Live video preview & multi-point facial ROI bounding boxes.
   - Filtered time-domain pulse waves & frequency power spectra (FFT).
   - Real-time physiological indicators (BPM per region, SNR, cross-patch coherence).
2. **Signal Extraction Pipelines**:
   - **Green Channel Filtering** (Fast optical baseline).
   - **POS (Plane-Orthogonal-to-Skin)**: Mitigates motion and lighting variance.
   - **CHROM (Chrominance-based rPPG)**: Color-difference pulse recovery.
3. **Multi-Region Consistency Analysis**:
   - Evaluates whether blood flows synchronously across facial quadrants. Synthetic artifacts and face swaps often desynchronize or warp patch rhythms.
4. **Leak-Free Machine Learning Evaluation**:
   - Implements **Subject-Independent GroupKFold Cross-Validation** via scikit-learn. Clips of the same subject (and deepfakes derived from that subject) never bleed across train/test splits.
   - Saved and versioned baseline models with `Joblib`.
5. **Principled Scientific Reporting**:
   - Clear visual distinction between preliminary raw signal analysis and validated model inferences.
   - Transparent documentation of rPPG vulnerabilities (compression, dark skin types, and 2024-2026 generative face reenactment models).

---

## Directory Structure

```
PulseGuard- AI/
├── app.py                      # Main Streamlit web application
├── generate_test_video.py      # Synthetic test video generator with rPPG modulation
├── requirements.txt            # Python dependencies
├── README.md                   # Setup guide and documentation
├── pulseguard/
│   ├── __init__.py
│   ├── models/
│   │   ├── classifier.py       # PulseGuardClassifier & GroupKFold pipeline
│   │   └── face_landmarker.task# MediaPipe Face Landmarker model asset
│   ├── pipeline/
│   │   ├── validator.py        # Video duration, FPS, and quality checks
│   │   └── trainer.py          # Training and benchmark dataset pipeline
│   ├── rppg/
│   │   ├── extractor.py        # MediaPipe ROI tracker (Forehead, Cheeks, Nose)
│   │   ├── processor.py        # Detrending, Butter filter, POS, CHROM, FFT
│   │   └── features.py         # Cross-spectral coherence and consistency
│   └── utils/
│       └── plots.py            # Matplotlib styling for waveforms and confusion matrix
├── data/
│   └── benchmark_dataset.csv   # Subject-grouped physiological benchmark dataset
├── models_saved/
│   └── pulseguard_svm_baseline.joblib # Serialized baseline SVM model
└── tests/
    └── test_pulseguard.py      # Unit & integration test suite
```

---

## Windows Setup Instructions

### 1. Prerequisites
- **Windows 10 or 11** (64-bit)
- **Python 3.10 or 3.11** installed and added to your Windows PATH.

### 2. Environment Setup (PowerShell)

Open PowerShell in the project directory:
```powershell
cd "C:\Users\jeyar\OneDrive\Attachments\Desktop\PulseGuard- AI"

# (Optional but recommended) Create a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Verify Tests
Run the unit test suite to verify signal extraction, FFT peak estimation, and classifier serialization:
```powershell
python tests/test_pulseguard.py
```
*(All tests should return `OK`.)*

### 4. Train / Verify Baseline Model
To generate the benchmark dataset and train the baseline SVM classifier:
```powershell
python -m pulseguard.pipeline.trainer
```

### 5. Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## Scientific Limitations & Ethical Caveats

- **Pulse Mimicry in Recent Deepfakes (2024–2026)**: While early deepfakes completely destroyed pulse signals, state-of-the-art face swaps created from high-resolution donor video may inherit donor pulse waveforms. Therefore, pulse detection alone does not constitute proof of authenticity; cross-region consistency and spectral coherence are mandatory.
- **Compression Artifacts**: Heavy lossy video compression (e.g., social media uploads) dampens fine micro-color variations, leading to false-positive fake classifications on real videos.
- **Illumination & Motion**: Sudden movements and non-diffuse flickering lights introduce noise orders of magnitude larger than capillary pulse signals.
- **Skin Tone Variations**: Hemoglobin absorption signals are naturally weaker on darker skin complexions (higher epidermal melanin). CHROM and POS algorithms help compensate, but adequate scene lux is critical.
