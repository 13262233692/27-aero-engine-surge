import numpy as np


class CompressorEKF:
    def __init__(
        self,
        dt: float = 0.01,
        process_noise_q: float = 1e-4,
        measurement_noise_r: float = 5e-3,
    ):
        self._dt = dt
        self._n_state = 4
        self._n_meas = 2

        self._x = np.array([25.0, 12.5, 0.0, 0.0], dtype=np.float64)

        self._P = np.diag([1.0, 0.5, 0.1, 0.1])

        self._Q = np.diag([
            process_noise_q,
            process_noise_q * 0.5,
            process_noise_q * 10,
            process_noise_q * 10,
        ])

        self._R = np.diag([measurement_noise_r, measurement_noise_r * 0.5])

        self._history_x = [self._x.copy()]
        self._max_history = 500

        self._predicted_trajectory = []
        self._prediction_horizon = 20

    def _state_transition(self, x: np.ndarray) -> np.ndarray:
        W, PR, dW, dPR = x
        damping_W = 0.15
        damping_PR = 0.10
        stiffness_W = 0.05
        stiffness_PR = 0.08

        ddW = -damping_W * dW - stiffness_W * (W - 25.0)
        ddPR = -damping_PR * dPR - stiffness_PR * (PR - 12.5)

        W_new = W + dW * self._dt
        PR_new = PR + dPR * self._dt
        dW_new = dW + ddW * self._dt
        dPR_new = dPR + ddPR * self._dt

        return np.array([W_new, PR_new, dW_new, dPR_new])

    def _jacobian_F(self, x: np.ndarray) -> np.ndarray:
        damping_W = 0.15
        damping_PR = 0.10
        stiffness_W = 0.05
        stiffness_PR = 0.08

        dt = self._dt
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        F[2, 0] = -stiffness_W * dt
        F[2, 2] = 1 - damping_W * dt
        F[3, 1] = -stiffness_PR * dt
        F[3, 3] = 1 - damping_PR * dt
        return F

    def predict(self):
        F = self._jacobian_F(self._x)
        self._x = self._state_transition(self._x)
        self._P = F @ self._P @ F.T + self._Q
        return self._x.copy()

    def update(self, z: np.ndarray):
        H = np.zeros((self._n_meas, self._n_state))
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        y = z - H @ self._x

        S = H @ self._P @ H.T + self._R
        K = self._P @ H.T @ np.linalg.inv(S)

        self._x = self._x + K @ y

        I_KH = np.eye(self._n_state) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T

        self._history_x.append(self._x.copy())
        if len(self._history_x) > self._max_history:
            self._history_x.pop(0)

        return self._x.copy()

    def predict_trajectory(self, n_steps: int = None):
        if n_steps is None:
            n_steps = self._prediction_horizon

        x_pred = self._x.copy()
        trajectory = [x_pred[:2].copy()]

        for _ in range(n_steps):
            x_pred = self._state_transition(x_pred)
            trajectory.append(x_pred[:2].copy())

        self._predicted_trajectory = trajectory
        return trajectory

    def compute_distance_to_surge(self, surge_line_func):
        W, PR = self._x[0], self._x[1]
        PR_surge = surge_line_func(W)
        margin = PR_surge - PR
        return margin

    def compute_bleed_valve_command(self, surge_line_func, threshold: float = 1.5):
        margin = self.compute_distance_to_surge(surge_line_func)

        if margin < threshold:
            severity = max(0.0, min(1.0, 1.0 - margin / threshold))
            command = {
                "active": True,
                "severity": severity,
                "valve_open_pct": severity * 100.0,
                "margin": margin,
                "message": f"SURGE IMMINENT - Margin: {margin:.2f} - BLEED VALVE {'OPEN' if severity > 0.5 else 'CRACKING'}",
            }
        else:
            command = {
                "active": False,
                "severity": 0.0,
                "valve_open_pct": 0.0,
                "margin": margin,
                "message": f"Normal - Margin: {margin:.2f}",
            }
        return command

    @property
    def state(self):
        return self._x.copy()

    @property
    def mass_flow(self):
        return self._x[0]

    @property
    def pressure_ratio(self):
        return self._x[1]

    @property
    def history(self):
        return [h.copy() for h in self._history_x]

    @property
    def predicted_trajectory(self):
        return [t.copy() for t in self._predicted_trajectory]

    def reset(self, initial_state=None):
        if initial_state is not None:
            self._x = np.array(initial_state, dtype=np.float64)
        else:
            self._x = np.array([25.0, 12.5, 0.0, 0.0], dtype=np.float64)
        self._P = np.diag([1.0, 0.5, 0.1, 0.1])
        self._history_x = [self._x.copy()]
        self._predicted_trajectory = []
