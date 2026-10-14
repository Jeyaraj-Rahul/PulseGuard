import numpy as np
import matplotlib.pyplot as plt
import io
from typing import Dict, List, Any, Optional

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
    roi_spectra: Dict[str, Dict[str, Any]],
    raw_signals: Optional[Dict[str, np.ndarray]] = None
):
    """
    Renders comparative raw signals, time-domain filtered pulse waveforms, and frequency power spectra across facial ROIs.
    """
    num_subplots = 3 if raw_signals is not None else 2
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 3.5 * num_subplots), dpi=100)
    fig.patch.set_facecolor('#0E1117')

    for ax in axes:
        ax.set_facecolor('#1E222A')

    colors = {
        "forehead": "#FF5722",
        "left_cheek": "#2196F3",
        "right_cheek": "#4CAF50",
        "nose": "#FFC107"
    }

    t = np.array(timestamps)
    max_t = t[-1] if len(t) > 0 else 0
    start_t = max(0, max_t - 6.0)
    time_mask = (t >= start_t)

    plot_idx = 0

    # 1. Raw Signals plot (if provided)
    if raw_signals is not None:
        ax_raw = axes[plot_idx]
        plot_idx += 1
        for roi, r_sig in raw_signals.items():
            c = colors.get(roi, "#9C27B0")
            r_slice = r_sig[time_mask] if len(r_sig) == len(t) else r_sig
            t_slice = t[time_mask] if len(r_sig) == len(t) else t[:len(r_slice)]
            ax_raw.plot(t_slice, r_slice, label=f"Raw {roi.replace('_', ' ').title()}", color=c, lw=1.2, alpha=0.85)

        ax_raw.set_title("Extracted Raw Mean RGB Intensity (Photoplethysmography Window)", color="white", fontsize=11, fontweight="bold")
        ax_raw.set_xlabel("Time (seconds)", color="#CCCCCC", fontsize=9)
        ax_raw.set_ylabel("Mean Intensity [0-255]", color="#CCCCCC", fontsize=9)
        ax_raw.tick_params(colors="#CCCCCC")
        ax_raw.grid(True, linestyle="--", alpha=0.2, color="white")
        ax_raw.legend(facecolor='#1E222A', edgecolor='#444444', labelcolor='white', loc='upper right')

    # 2. Filtered Pulse Waveforms plot
    ax_time = axes[plot_idx]
    plot_idx += 1
    for roi, sig in roi_signals.items():
        c = colors.get(roi, "#9C27B0")
        sig_slice = sig[time_mask] if len(sig) == len(t) else sig
        t_slice = t[time_mask] if len(sig) == len(t) else t[:len(sig_slice)]
        ax_time.plot(t_slice, sig_slice, label=roi.replace('_', ' ').title(), color=c, lw=1.6, alpha=0.9)

    ax_time.set_title("Filtered rPPG Pulse Waveforms (Detrended & Bandpass 0.7-3.0 Hz)", color="white", fontsize=11, fontweight="bold")
    ax_time.set_xlabel("Time (seconds)", color="#CCCCCC", fontsize=9)
    ax_time.set_ylabel("Normalized Amplitude", color="#CCCCCC", fontsize=9)
    ax_time.tick_params(colors="#CCCCCC")
    ax_time.grid(True, linestyle="--", alpha=0.2, color="white")
    ax_time.legend(facecolor='#1E222A', edgecolor='#444444', labelcolor='white', loc='upper right')

    # 3. Power Spectral Density (FFT)
    ax_freq = axes[plot_idx]
    for roi, spec in roi_spectra.items():
        c = colors.get(roi, "#9C27B0")
        freqs = spec.get("freqs", np.array([]))
        power = spec.get("power", np.array([]))
        bpm = spec.get("bpm", 0.0)
        snr = spec.get("snr_db", -20.0)

        if len(freqs) > 0 and len(power) > 0:
            mask = (freqs >= 0.5) & (freqs <= 3.5)
            p_slice = power[mask]
            f_slice = freqs[mask]
            max_p = np.max(p_slice) if len(p_slice) > 0 else 0.0
            if max_p > 1e-12:
                norm_p = p_slice / max_p
            else:
                norm_p = np.zeros_like(p_slice)
            label_text = f"{roi.replace('_', ' ').title()} ({bpm:.0f} BPM, {snr:.1f} dB)" if bpm > 0 else f"{roi.replace('_', ' ').title()} (No Peak)"
            ax_freq.plot(f_slice * 60.0, norm_p, label=label_text, color=c, lw=1.5, alpha=0.85)

    ax_freq.set_title("Pulse Power Spectrum (FFT Cardiac Band)", color="white", fontsize=11, fontweight="bold")
    ax_freq.set_xlabel("Frequency (BPM)", color="#CCCCCC", fontsize=9)
    ax_freq.set_ylabel("Normalized Power", color="#CCCCCC", fontsize=9)
    ax_freq.tick_params(colors="#CCCCCC")
    ax_freq.grid(True, linestyle="--", alpha=0.2, color="white")
    ax_freq.legend(facecolor='#1E222A', edgecolor='#444444', labelcolor='white', loc='upper right')

    plt.tight_layout()
    return fig
