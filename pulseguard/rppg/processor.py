import numpy as np
from scipy import signal
from typing import Dict, Tuple, Optional, Union, Any

class SignalProcessor:
    """
    Digital signal processing pipeline for rPPG signals:
    - Normalization & detrending (scipy.signal.detrend)
    - Zero-phase Butterworth bandpass filtering (0.7 Hz - 3.0 Hz, 42 - 180 BPM)
    - rPPG extraction methods: Green-channel baseline, CHROM (Chrominance), POS (Plane-Orthogonal-to-Skin)
    - FFT spectral estimation and SNR calculation
    """

    def __init__(
        self,
        low_cut_hz: float = 0.7,   # ~42 BPM
        high_cut_hz: float = 3.0,  # ~180 BPM
        filter_order: int = 4
    ):
        self.low_cut_hz = low_cut_hz
        self.high_cut_hz = high_cut_hz
        self.filter_order = filter_order

    def detrend(self, x: np.ndarray) -> np.ndarray:
        """Subtract linear trend to remove slow baseline illumination shifts."""
        if len(x) < 2:
            return x
        return signal.detrend(x, type='linear')

    def bandpass_filter(self, x: np.ndarray, fps: float) -> np.ndarray:
        """
        Zero-phase bandpass filter using Second-Order Sections (SOS) for numerical stability.
        Filter bounds: [low_cut_hz, high_cut_hz]
        """
        if len(x) < 15:
            return x

        nyq = 0.5 * fps
        low = max(0.01, self.low_cut_hz / nyq)
        high = min(0.99, self.high_cut_hz / nyq)

        if low >= high:
            return x - np.mean(x)

        sos = signal.butter(self.filter_order, [low, high], btype='bandpass', output='sos')
        filtered = signal.sosfiltfilt(sos, x)
        return filtered

    def extract_green_signal(self, g_channel: np.ndarray, fps: float) -> np.ndarray:
        """
        Standard Green-channel pulse extraction:
        1. Temporal normalization (G / mean(G) - 1)
        2. Linear detrending
        3. Zero-phase Butterworth bandpass filter
        """
        g_arr = np.array(g_channel, dtype=float)
        mean_val = np.mean(g_arr)
        if mean_val > 1e-6:
            norm_g = (g_arr / mean_val) - 1.0
        else:
            norm_g = g_arr - mean_val

        detrended = self.detrend(norm_g)
        filtered = self.bandpass_filter(detrended, fps)
        return filtered

    def extract_chrom_signal(self, r: np.ndarray, g: np.ndarray, b: np.ndarray, fps: float) -> np.ndarray:
        """
        Chrominance-based rPPG method (De Haan & Jeanne, 2013).
        X = 3R - 2G
        Y = 1.5R + G - 1.5B
        bvp = X - alpha * Y, where alpha = std(X) / std(Y)
        """
        r_arr = np.array(r, dtype=float)
        g_arr = np.array(g, dtype=float)
        b_arr = np.array(b, dtype=float)

        # Normalize by mean
        r_norm = r_arr / (np.mean(r_arr) + 1e-6)
        g_norm = g_arr / (np.mean(g_arr) + 1e-6)
        b_norm = b_arr / (np.mean(b_arr) + 1e-6)

        x_s = 3.0 * r_norm - 2.0 * g_norm
        y_s = 1.5 * r_norm + g_norm - 1.5 * b_norm

        std_y = np.std(y_s)
        if std_y > 1e-6:
            alpha = np.std(x_s) / std_y
        else:
            alpha = 1.0

        s = x_s - alpha * y_s
        detrended = self.detrend(s)
        filtered = self.bandpass_filter(detrended, fps)
        return filtered

    def extract_pos_signal(self, r: np.ndarray, g: np.ndarray, b: np.ndarray, fps: float) -> np.ndarray:
        """
        Plane-Orthogonal-to-Skin (POS) algorithm (Wang et al., 2017).
        Projects normalized RGB channels onto a plane orthogonal to the skin tone vector.
        """
        r_arr = np.array(r, dtype=float)
        g_arr = np.array(g, dtype=float)
        b_arr = np.array(b, dtype=float)

        r_norm = r_arr / (np.mean(r_arr) + 1e-6)
        g_norm = g_arr / (np.mean(g_arr) + 1e-6)
        b_norm = b_arr / (np.mean(b_arr) + 1e-6)

        # S matrix projection
        s1 = g_norm - b_norm
        s2 = g_norm + b_norm - 2.0 * r_norm

        std_s2 = np.std(s2)
        if std_s2 > 1e-6:
            alpha = np.std(s1) / std_s2
        else:
            alpha = 1.0

        pos_signal = s1 + alpha * s2
        detrended = self.detrend(pos_signal)
        filtered = self.bandpass_filter(detrended, fps)
        return filtered

    def compute_spectrum(self, sig: np.ndarray, fps: float) -> Dict[str, Any]:
        """
        Compute Fast Fourier Transform (FFT) power spectral density,
        identify peak heart rate frequency in Hz and BPM, and calculate Signal-to-Noise Ratio (SNR).
        """
        n = len(sig)
        if n < 8:
            return {
                "bpm": 0.0,
                "peak_hz": 0.0,
                "snr_db": 0.0,
                "freqs": np.array([]),
                "power": np.array([])
            }

        # Apply Hanning window to reduce spectral leakage
        windowed = sig * np.hanning(n)

        # Zero-pad for finer frequency bin resolution (at least 2048 points)
        n_fft = max(2048, 2 ** int(np.ceil(np.log2(n))))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)
        fft_vals = np.fft.rfft(windowed, n=n_fft)
        power = np.abs(fft_vals) ** 2

        # In-band mask for valid physiological heart rates
        band_mask = (freqs >= self.low_cut_hz) & (freqs <= self.high_cut_hz)
        if not np.any(band_mask):
            return {
                "bpm": 0.0,
                "peak_hz": 0.0,
                "snr_db": 0.0,
                "freqs": freqs,
                "power": power
            }

        # Check for zero or flat signal (variance near zero)
        if np.std(sig) < 1e-7 or np.all(sig == 0):
            return {
                "bpm": 0.0,
                "peak_hz": 0.0,
                "snr_db": -20.0,
                "freqs": freqs,
                "power": power
            }

        band_power = power[band_mask]
        band_freqs = freqs[band_mask]
        max_power = float(np.max(band_power))

        # If in-band power is flat or virtually zero, no pulse detected
        if max_power < 1e-12:
            return {
                "bpm": 0.0,
                "peak_hz": 0.0,
                "snr_db": -20.0,
                "freqs": freqs,
                "power": power
            }

        peak_idx = np.argmax(band_power)
        peak_hz = float(band_freqs[peak_idx])
        peak_bpm = float(peak_hz * 60.0)

        # Signal-to-Noise Ratio (SNR) in dB within the cardiac band
        # Cardiac signal power: narrow region around peak (+/- 0.15 Hz) & first harmonic
        harmonic_hz = peak_hz * 2.0
        signal_mask = ((band_freqs >= peak_hz - 0.15) & (band_freqs <= peak_hz + 0.15))
        if harmonic_hz <= self.high_cut_hz:
            signal_mask |= ((band_freqs >= harmonic_hz - 0.15) & (band_freqs <= harmonic_hz + 0.15))

        sig_power = float(np.sum(band_power[signal_mask]))
        noise_power = float(np.sum(band_power[~signal_mask]))

        if sig_power <= 1e-12:
            snr_db = -20.0
        elif noise_power > 1e-12:
            snr_db = float(10.0 * np.log10(sig_power / noise_power))
        else:
            snr_db = 15.0

        return {
            "bpm": round(peak_bpm, 1),
            "peak_hz": round(peak_hz, 3),
            "snr_db": round(snr_db, 2),
            "freqs": freqs,
            "power": power
        }
