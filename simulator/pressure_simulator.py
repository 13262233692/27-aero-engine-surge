import numpy as np
import time


class PressureSimulator:
    def __init__(
        self,
        sample_rate: int = 100000,
        n_sensors: int = 1,
        base_pressure_kpa: float = 101.325,
        noise_level: float = 0.5,
    ):
        self._sample_rate = sample_rate
        self._n_sensors = n_sensors
        self._base_pressure = base_pressure_kpa
        self._noise_level = noise_level
        self._blade_pass_freq = 3500.0
        self._n_blades = 24
        self._rotor_rpm = 8750
        self._surge_onset_time = 15.0
        self._surge_intensity = 0.0
        self._running = False
        self._elapsed = 0.0
        self._W_steady = 25.0
        self._PR_steady = 12.5
        self._current_W = self._W_steady
        self._current_PR = self._PR_steady

    def generate_batch(self, duration: float = 0.01):
        n_samples = int(duration * self._sample_rate)
        t = np.arange(n_samples) / self._sample_rate + self._elapsed

        signal = np.zeros(n_samples, dtype=np.float32)

        bpf = self._blade_pass_freq
        signal += 2.0 * np.sin(2 * np.pi * bpf * t)
        signal += 1.0 * np.sin(2 * np.pi * bpf * 2 * t)
        signal += 0.5 * np.sin(2 * np.pi * bpf * 3 * t + 0.3)

        signal += 0.8 * np.sin(2 * np.pi * 12000 * t)
        signal += 0.3 * np.sin(2 * np.pi * 18000 * t)

        signal += self._noise_level * np.random.randn(n_samples).astype(np.float32)

        elapsed_abs = self._elapsed + t
        surge_mask = elapsed_abs > self._surge_onset_time

        if np.any(surge_mask):
            surge_t = elapsed_abs[surge_mask] - self._surge_onset_time
            self._surge_intensity = min(1.0, self._surge_intensity + 0.0001)

            precursor_freq = 800.0
            signal[surge_mask] += (
                3.0 * self._surge_intensity
                * np.sin(2 * np.pi * precursor_freq * surge_t)
                * (1 + 0.5 * np.sin(2 * np.pi * 3.5 * surge_t))
            )

            modal_freq = 2200.0
            signal[surge_mask] += (
                1.5 * self._surge_intensity
                * np.sin(2 * np.pi * modal_freq * surge_t)
                * np.exp(-0.5 * surge_t % 0.3)
            )

            burst_mask = (surge_t % 0.5) < 0.02
            combined_mask = np.zeros(n_samples, dtype=bool)
            combined_mask[surge_mask] = burst_mask
            signal[combined_mask] += 8.0 * self._surge_intensity * np.random.randn(np.sum(combined_mask)).astype(np.float32)

        timestamps = t.copy()

        W, PR = self._generate_compressor_state(duration)

        self._elapsed += duration

        return signal, timestamps, W, PR

    def _generate_compressor_state(self, duration: float):
        elapsed = self._elapsed + duration
        noise_W = np.random.randn() * 0.15
        noise_PR = np.random.randn() * 0.08

        if elapsed < self._surge_onset_time:
            ramp_W = -0.05 * elapsed
            ramp_PR = 0.03 * elapsed
            self._current_W = self._W_steady + ramp_W + noise_W
            self._current_PR = self._PR_steady + ramp_PR + noise_PR
        else:
            surge_t = elapsed - self._surge_onset_time
            drift_rate = min(1.0, surge_t * 0.04)

            self._current_W = (
                self._W_steady
                - 0.05 * self._surge_onset_time
                - drift_rate * 8.0
                + 0.5 * np.sin(2 * np.pi * 0.8 * surge_t) * drift_rate
                + noise_W * (1.0 + drift_rate * 3.0)
            )

            self._current_PR = (
                self._PR_steady
                + 0.03 * self._surge_onset_time
                + drift_rate * 2.5
                + 0.3 * np.sin(2 * np.pi * 1.2 * surge_t) * drift_rate
                + noise_PR * (1.0 + drift_rate * 2.0)
            )

        return self._current_W, self._current_PR

    def generate_continuous(self, batch_duration: float = 0.01, callback=None):
        self._running = True
        self._elapsed = 0.0
        self._surge_intensity = 0.0
        self._current_W = self._W_steady
        self._current_PR = self._PR_steady

        try:
            while self._running:
                data, ts, W, PR = self.generate_batch(batch_duration)
                if callback:
                    callback(data, ts, W, PR)
                sleep_time = batch_duration * 0.8
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            self._running = False

    def stop(self):
        self._running = False

    @property
    def running(self):
        return self._running

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def surge_onset_time(self):
        return self._surge_onset_time

    @property
    def current_compressor_state(self):
        return self._current_W, self._current_PR
