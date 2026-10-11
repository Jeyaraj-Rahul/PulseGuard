import numpy as np
import matplotlib.pyplot as plt
import io
from typing import Dict, List, Any

def create_confusion_matrix_figure(cm: List[List[int]], class_names: List[str] = ["Manipulated", "Real"]):
    """
    Renders a formatted Matplotlib figure of the confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.8), dpi=100)
    cm_arr = np.array(cm)

    im = ax.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="GroupKFold Confusion Matrix",
        ylabel="True Class",
        xlabel="Predicted Class"
    )

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, format(cm_arr[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontweight='bold', fontsize=12
            )

    plt.tight_layout()
    return fig

def create_waveform_and_spectrum_figure(
    timestamps: np.ndarray,
    roi_signals: Dict[str, np.ndarray],
    roi_spectra: Dict[str, Dict[str, Any]]
):
    """
    Renders comparative time-domain filtered pulse waveforms and frequency power spectra across facial ROIs.
    """
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor('#0E1117')
    ax_time.set_facecolor('#1E222A')
    ax_freq.set_facecolor('#1E222A')

    colors = {
        "forehead": "#FF5722",
        "left_cheek": "#2196F3",
        "right_cheek": "#4CAF50",
        "nose": "#FFC107"
    }

    # Plot waveforms (last 6 seconds or full clip)
    t = np.array(timestamps)
    max_t = t[-1] if len(t) > 0 else 0
    start_t = max(0, max_t - 6.0)
    time_mask = (t >= start_t)

    for roi, sig in roi_signals.items():
        c = colors.get(roi, "#9C27B0")
        sig_slice = sig[time_mask] if len(sig) == len(t) else sig
        t_slice = t[time_mask] if len(sig) == len(t) else t[:len(sig_slice)]
        ax_time.plot(t_slice, sig_slice, label=roi.replace('_', ' ').title(), color=c, lw=1.6, alpha=0.9)

    ax_time.set_title("Filtered rPPG Pulse Waveforms (Selected Window)", color="white", fontsize=11, fontweight="bold")
    ax_time.set_xlabel("Time (seconds)", color="#CCCCCC", fontsize=9)
    ax_time.set_ylabel("Normalized Amplitude", color="#CCCCCC", fontsize=9)
    ax_time.tick_params(colors="#CCCCCC")
    ax_time.grid(True, linestyle="--", alpha=0.2, color="white")
    ax_time.legend(facecolor='#1E222A', edgecolor='#444444', labelcolor='white', loc='upper right')

    # Plot Power Spectral Density (FFT) in cardiac range [0.5 - 3.5 Hz]
    for roi, spec in roi_spectra.items():
        c = colors.get(roi, "#9C27B0")
        freqs = spec.get("freqs", np.array([]))
        power = spec.get("power", np.array([]))
        bpm = spec.get("bpm", 0.0)

        if len(freqs) > 0 and len(power) > 0:
            mask = (freqs >= 0.5) & (freqs <= 3.5)
            # Normalize power for visual overlay
            p_slice = power[mask]
            f_slice = freqs[mask]
            norm_p = p_slice / (np.max(p_slice) + 1e-9)
            ax_freq.plot(f_slice * 60.0, norm_p, label=f"{roi.replace('_', ' ').title()} ({bpm:.0f} BPM)", color=c, lw=1.5, alpha=0.85)

    ax_freq.set_title("Pulse Power Spectrum (FFT)", color="white", fontsize=11, fontweight="bold")
    ax_freq.set_xlabel("Frequency (BPM)", color="#CCCCCC", fontsize=9)
    ax_freq.set_ylabel("Normalized Power", color="#CCCCCC", fontsize=9)
    ax_freq.tick_params(colors="#CCCCCC")
    ax_freq.grid(True, linestyle="--", alpha=0.2, color="white")
    ax_freq.legend(facecolor='#1E222A', edgecolor='#444444', labelcolor='white', loc='upper right')

    plt.tight_layout()
    return fig
