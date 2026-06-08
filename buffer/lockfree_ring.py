import ctypes
import numpy as np
from threading import current_thread
import time


class LockFreeRingBuffer:
    def __init__(self, capacity: int, dtype=np.float32):
        self._capacity = capacity
        self._dtype = dtype
        self._buffer = np.zeros(capacity, dtype=dtype)
        self._timestamps = np.zeros(capacity, dtype=np.float64)
        self._write_idx = ctypes.c_ulonglong(0)
        self._read_idx = ctypes.c_ulonglong(0)
        self._total_written = ctypes.c_ulonglong(0)
        self._sample_rate = 0

    @property
    def capacity(self):
        return self._capacity

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value

    def write(self, data: np.ndarray, timestamps: np.ndarray = None):
        n = len(data)
        if n > self._capacity:
            data = data[-self._capacity:]
            n = self._capacity
            if timestamps is not None:
                timestamps = timestamps[-self._capacity:]

        pos = self._write_idx.value
        end = pos + n

        if end <= self._capacity:
            self._buffer[pos:end] = data
            if timestamps is not None:
                self._timestamps[pos:end] = timestamps
        else:
            first = self._capacity - pos
            self._buffer[pos:] = data[:first]
            self._buffer[:n - first] = data[first:]
            if timestamps is not None:
                self._timestamps[pos:] = timestamps[:first]
                self._timestamps[:n - first] = timestamps[first:]

        self._write_idx.value = end % self._capacity
        ctypes.atomic_add = None
        old_total = self._total_written.value
        self._total_written.value = old_total + n

    def read_latest(self, n_samples: int):
        total = self._total_written.value
        if total == 0:
            return np.array([], dtype=self._dtype), np.array([], dtype=np.float64)

        n_samples = min(n_samples, total)
        end_pos = self._write_idx.value
        start_pos = (end_pos - n_samples) % self._capacity

        if start_pos < end_pos:
            return (
                self._buffer[start_pos:end_pos].copy(),
                self._timestamps[start_pos:end_pos].copy(),
            )
        else:
            data = np.concatenate([
                self._buffer[start_pos:],
                self._buffer[:end_pos],
            ])
            ts = np.concatenate([
                self._timestamps[start_pos:],
                self._timestamps[:end_pos],
            ])
            return data, ts

    def read_slice_by_time(self, start_time: float, end_time: float):
        total = self._total_written.value
        if total == 0:
            return np.array([], dtype=self._dtype), np.array([], dtype=np.float64)

        end_pos = self._write_idx.value
        n_avail = min(total, self._capacity)
        start_pos = (end_pos - n_avail) % self._capacity

        if start_pos < end_pos:
            ts_slice = self._timestamps[start_pos:end_pos]
            mask = (ts_slice >= start_time) & (ts_slice < end_time)
            return (
                self._buffer[start_pos:end_pos][mask].copy(),
                ts_slice[mask].copy(),
            )
        else:
            ts_all = np.concatenate([
                self._timestamps[start_pos:],
                self._timestamps[:end_pos],
            ])
            mask = (ts_all >= start_time) & (ts_all < end_time)
            data_all = np.concatenate([
                self._buffer[start_pos:],
                self._buffer[:end_pos],
            ])
            return data_all[mask].copy(), ts_all[mask].copy()

    def get_latest_second(self):
        total = self._total_written.value
        if total == 0:
            return np.array([], dtype=self._dtype), np.array([], dtype=np.float64)

        end_pos = self._write_idx.value
        if end_pos == 0 and total > 0:
            latest_ts = self._timestamps[-1]
        elif total > 0:
            latest_ts = self._timestamps[end_pos - 1]
        else:
            return np.array([], dtype=self._dtype), np.array([], dtype=np.float64)

        return self.read_slice_by_time(latest_ts - 1.0, latest_ts)

    @property
    def utilization(self):
        total = self._total_written.value
        return min(100, int((min(total, self._capacity) / self._capacity) * 100))

    @property
    def total_written(self):
        return self._total_written.value

    @property
    def latest_timestamp(self):
        end_pos = self._write_idx.value
        if end_pos == 0:
            if self._total_written.value > 0:
                return float(self._timestamps[-1])
            return 0.0
        return float(self._timestamps[end_pos - 1])
