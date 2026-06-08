import numpy as np


class SurgeLine:
    def __init__(
        self,
        W_design: float = 25.0,
        PR_design: float = 12.5,
        W_surge_min: float = 10.0,
    ):
        self._W_design = W_design
        self._PR_design = PR_design
        self._W_surge_min = W_surge_min
        self._coeffs = self._calibrate()

    def _calibrate(self):
        W_s = self._W_surge_min
        W_d = self._W_design
        PR_d = self._PR_design

        a = -0.018
        b = 0.6
        c = PR_d - a * W_d ** 2 - b * W_d

        return (a, b, c)

    def __call__(self, W: float) -> float:
        a, b, c = self._coeffs
        if W < self._W_surge_min:
            return a * self._W_surge_min ** 2 + b * self._W_surge_min + c
        return a * W ** 2 + b * W + c

    def evaluate_array(self, W_array: np.ndarray) -> np.ndarray:
        a, b, c = self._coeffs
        W_clipped = np.clip(W_array, self._W_surge_min, None)
        return a * W_clipped ** 2 + b * W_clipped + c

    def get_surge_line_points(self, n_points: int = 100, W_range=None):
        if W_range is None:
            W_range = (self._W_surge_min, self._W_design * 1.3)
        W = np.linspace(W_range[0], W_range[1], n_points)
        PR = self.evaluate_array(W)
        return W, PR

    def is_below_surge_line(self, W: float, PR: float) -> bool:
        return PR >= self(W)

    def surge_margin(self, W: float, PR: float) -> float:
        PR_surge = self(W)
        if PR_surge <= 0:
            return float('inf')
        return (PR_surge - PR) / PR_surge * 100.0

    def get_compressor_map_speed_lines(self, n_speeds: int = 6, n_points: int = 50):
        W_min = self._W_surge_min
        W_max = self._W_design * 1.4
        speed_lines = []

        for i in range(n_speeds):
            frac = (i + 1) / n_speeds
            N_pct = 60 + frac * 40

            W_line = np.linspace(W_min + 2, W_max, n_points)

            a, b, c = self._coeffs
            PR_peak = a * W_line ** 2 + b * W_line + c

            PR_shift = (1.0 - frac) * 4.0
            PR_line = PR_peak - PR_shift

            W_eff = W_line * (0.85 + 0.15 * frac)
            PR_eff = PR_line * (0.88 + 0.12 * frac)

            speed_lines.append({
                "W": W_line,
                "PR": PR_line,
                "W_eff": W_eff,
                "PR_eff": PR_eff,
                "N_pct": N_pct,
            })

        return speed_lines

    @property
    def coeffs(self):
        return self._coeffs

    @property
    def W_surge_min(self):
        return self._W_surge_min

    @property
    def W_design(self):
        return self._W_design

    @property
    def PR_design(self):
        return self._PR_design
