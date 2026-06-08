import numpy as np
from scipy.signal import morlet2, cwt as scipy_cwt
from scipy.fft import fftfreq


class CWTAnalyzer:
    def __init__(
        self,
        sample_rate: int,
        freq_min: float = 500.0,
        freq_max: float = 45000.0,
        n_scales: int = 128,
        log_scale: bool = True,
    ):
        self._sample_rate = sample_rate
        self._freq_min = freq_min
        self._freq_max = min(freq_max, sample_rate / 2.0)
        self._n_scales = n_scales
        self._log_scale = log_scale
        self._width = 6.0
        self._scales = self._compute_scales()
        self._freqs = self._width * sample_rate / (2 * np.pi * self._scales)

    def _compute_scales(self):
        if self._log_scale:
            return np.logspace(
                np.log10(self._width * self._sample_rate / (2 * np.pi * self._freq_min)),
                np.log10(self._width * self._sample_rate / (2 * np.pi * self._freq_max)),
                self._n_scales,
            )
        else:
            return np.linspace(
                self._width * self._sample_rate / (2 * np.pi * self._freq_max),
                self._width * self._sample_rate / (2 * np.pi * self._freq_min),
                self._n_scales,
            )

    def compute(self, data: np.ndarray):
        if len(data) < 64:
            return None, None, None

        n = len(data)
        cwt_matrix = np.zeros((len(self._scales), n), dtype=np.complex128)

        for i, s in enumerate(self._scales):
            wavelet = morlet2(n, s, w=self._width)
            cwt_matrix[i, :] = np.convolve(data, wavelet, mode='same')

        power = np.abs(cwt_matrix) ** 2

        power_log = 10 * np.log10(power + 1e-12)

        power_log = np.clip(power_log, np.percentile(power_log, 1), np.percentile(power_log, 99))

        return power_log, self._freqs, self._scales

    def compute_heatmap_data(self, data: np.ndarray, time_offset: float = 0.0):
        power_log, freqs, scales = self.compute(data)
        if power_log is None:
            return None

        n = len(data)
        duration = n / self._sample_rate
        times = np.linspace(time_offset, time_offset + duration, n)

        return {
            "power": power_log,
            "freqs": freqs,
            "times": times,
            "freq_min": self._freq_min,
            "freq_max": self._freq_max,
            "duration": duration,
        }

    def downsample_heatmap(self, heatmap_data: dict, target_time_bins: int = 256, target_freq_bins: int = 64):
        if heatmap_data is None:
            return None

        power = heatmap_data["power"]
        freqs = heatmap_data["freqs"]
        times = heatmap_data["times"]

        n_time = power.shape[1]
        n_freq = power.shape[0]

        t_step = max(1, n_time // target_time_bins)
        f_step = max(1, n_freq // target_freq_bins)

        power_ds = power[::f_step, ::t_step]
        freqs_ds = freqs[::f_step]
        times_ds = times[::t_step]

        return {
            "power": power_ds,
            "freqs": freqs_ds,
            "times": times_ds,
            "freq_min": heatmap_data["freq_min"],
            "freq_max": heatmap_data["freq_max"],
            "duration": heatmap_data["duration"],
        }

    @property
    def scales(self):
        return self._scales

    @property
    def freqs(self):
        return self._freqs

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def n_scales(self):
        return self._n_scales
