from __future__ import annotations

from collections import deque

import cv2
import numpy as np


def update_mode_history(history: dict[str, deque[float]], dx: float, dy: float, da: float) -> None:
    da_deg = float(np.degrees(da))
    history["dx"].append(float(dx))
    history["dy"].append(float(dy))
    history["da_deg"].append(da_deg)
    history["trans_mag"].append(float(np.hypot(dx, dy)))
    history["rot_mag_deg"].append(abs(da_deg))


def estimate_motion_state(
    dx: float, dy: float, da: float, history: dict[str, deque[float]]
) -> tuple[str, dict[str, float]]:
    if len(history["dx"]) < 3:
        return "normal", {"confidence": 0.0}

    dx_hist = np.fromiter(history["dx"], dtype=np.float32)
    dy_hist = np.fromiter(history["dy"], dtype=np.float32)
    da_hist_deg = np.fromiter(history["da_deg"], dtype=np.float32)
    trans_hist = np.fromiter(history["trans_mag"], dtype=np.float32)
    rot_hist_deg = np.fromiter(history["rot_mag_deg"], dtype=np.float32)

    def sign_change_ratio(values: np.ndarray, eps: float) -> float:
        if len(values) < 2:
            return 0.0
        prev = values[:-1]
        curr = values[1:]
        valid = (np.abs(prev) > eps) & (np.abs(curr) > eps)
        if not np.any(valid):
            return 0.0
        return float(np.mean(np.sign(prev[valid]) != np.sign(curr[valid])))

    sign_changes = (
        sign_change_ratio(dx_hist, 0.35) + sign_change_ratio(dy_hist, 0.35) + sign_change_ratio(da_hist_deg, 0.05)
    ) / 3.0
    if len(dx_hist) > 1:
        jerk = float(
            np.mean(np.abs(np.diff(dx_hist)))
            + np.mean(np.abs(np.diff(dy_hist)))
            + 2.0 * np.mean(np.abs(np.diff(da_hist_deg)))
        )
    else:
        jerk = 0.0

    moving = trans_hist > 0.5
    if np.any(moving):
        angles = np.arctan2(dy_hist[moving], dx_hist[moving])
        direction_consistency = float(np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles))))
    else:
        direction_consistency = 1.0

    mean_trans = float(np.mean(trans_hist))
    max_trans = float(np.max(trans_hist))
    std_trans = float(np.std(trans_hist))
    mean_rot_deg = float(np.mean(rot_hist_deg))
    max_rot_deg = float(np.max(rot_hist_deg))

    violent_score = 0
    violent_score += int(max_trans > 8.0 or max_rot_deg > 0.8)
    violent_score += int(jerk > 5.0)
    violent_score += int(sign_changes > 0.45)
    violent_score += int(std_trans > 3.0)
    violent_score += int(direction_consistency < 0.45 and mean_trans > 2.0)

    large_sway_score = 0
    large_sway_score += int(mean_trans > 2.5 or mean_rot_deg > 0.25)
    large_sway_score += int(max_trans > 5.0 or max_rot_deg > 0.6)
    large_sway_score += int(direction_consistency > 0.65)
    large_sway_score += int(sign_changes < 0.35)
    large_sway_score += int(jerk < 5.5)

    if violent_score >= 3 or (violent_score >= 2 and max_trans > 10.0):
        mode = "violent"
        confidence = violent_score / 5.0
    elif large_sway_score >= 4:
        mode = "large_sway"
        confidence = large_sway_score / 5.0
    else:
        mode = "normal"
        confidence = max(0.0, 1.0 - mean_trans / 8.0)

    return mode, {
        "confidence": float(confidence),
        "mean_trans": mean_trans,
        "max_trans": max_trans,
        "std_trans": std_trans,
        "mean_rot_deg": mean_rot_deg,
        "max_rot_deg": max_rot_deg,
        "jerk": jerk,
        "sign_changes": float(sign_changes),
        "direction_consistency": direction_consistency,
    }


def get_mode_params(mode: str) -> dict[str, float]:
    presets = {
        "violent": {
            "alpha": 0.70,
            "max_translation": 6.0,
            "max_rotation": float(np.deg2rad(0.75)),
            "roi_ratio": 0.60,
            "max_corners": 120.0,
            "gain": 0.55,
            "rotation_gain": 0.50,
        },
        "large_sway": {
            "alpha": 0.88,
            "max_translation": 10.0,
            "max_rotation": float(np.deg2rad(1.10)),
            "roi_ratio": 0.70,
            "max_corners": 180.0,
            "gain": 0.82,
            "rotation_gain": 0.80,
        },
        "normal": {
            "alpha": 0.80,
            "max_translation": 8.0,
            "max_rotation": float(np.deg2rad(0.95)),
            "roi_ratio": 0.74,
            "max_corners": 160.0,
            "gain": 0.72,
            "rotation_gain": 0.70,
        },
    }
    return dict(presets[mode])


def apply_gain_schedule(dx: float, dy: float, da: float, params: dict[str, float]) -> tuple[float, float, float]:
    dx *= params["gain"]
    dy *= params["gain"]
    da *= params["rotation_gain"]
    dx = float(np.clip(dx, -params["max_translation"], params["max_translation"]))
    dy = float(np.clip(dy, -params["max_translation"], params["max_translation"]))
    da = float(np.clip(da, -params["max_rotation"], params["max_rotation"]))
    return dx, dy, da


class SimpleStabilizer:
    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: float = 20.0,
        block_size: int = 3,
        lk_win_size: tuple[int, int] = (21, 21),
        lk_max_level: int = 3,
        smoothing_alpha: float = 0.8,
    ) -> None:
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.lk_params = dict(
            winSize=lk_win_size,
            maxLevel=lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.downsample_scale = 0.5
        self.max_translation = 12.0
        self.max_rotation = float(np.deg2rad(1.5))
        self.pad = 24
        self.roi_ratio = 0.7
        self.estimator_roi_ratio = 0.7
        self.estimator_max_corners = min(self.max_corners, 160)
        self.alpha_offset = smoothing_alpha - 0.8
        self.prev_gray: np.ndarray | None = None
        self.trajectory = np.zeros(3, dtype=np.float32)
        self.smooth_trajectory = np.zeros(3, dtype=np.float32)
        self.mode = "normal"
        self.active_params = self._resolve_mode_params(self.mode)
        self.mode_history: dict[str, deque[float]] = {
            "dx": deque(maxlen=8),
            "dy": deque(maxlen=8),
            "da_deg": deque(maxlen=8),
            "trans_mag": deque(maxlen=8),
            "rot_mag_deg": deque(maxlen=8),
        }
        self.pending_mode: str | None = None
        self.pending_count = 0
        self.last_motion_state: dict[str, float] = {"confidence": 0.0}

    def _resolve_mode_params(self, mode: str) -> dict[str, float]:
        params = get_mode_params(mode)
        params["alpha"] = float(np.clip(params["alpha"] + self.alpha_offset, 0.60, 0.92))
        params["max_translation"] = min(params["max_translation"], self.max_translation)
        params["max_rotation"] = min(params["max_rotation"], self.max_rotation)
        params["roi_ratio"] = min(params["roi_ratio"], self.roi_ratio)
        params["max_corners"] = min(params["max_corners"], float(self.max_corners))
        return params

    def estimate_transform(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[float, float, float]:
        scale = float(self.downsample_scale)
        prev_small = cv2.resize(prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        h_s, w_s = prev_small.shape
        roi_w = max(32, int(w_s * self.estimator_roi_ratio))
        roi_h = max(32, int(h_s * self.estimator_roi_ratio))
        x0 = (w_s - roi_w) // 2
        y0 = (h_s - roi_h) // 2
        prev_roi = prev_small[y0 : y0 + roi_h, x0 : x0 + roi_w]
        prev_pts = cv2.goodFeaturesToTrack(
            prev_roi,
            maxCorners=self.estimator_max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
        )
        if prev_pts is None or len(prev_pts) < 8:
            return 0.0, 0.0, 0.0

        prev_pts[:, :, 0] += x0
        prev_pts[:, :, 1] += y0
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_small, curr_small, prev_pts, None, **self.lk_params)
        if curr_pts is None or status is None:
            return 0.0, 0.0, 0.0

        status = status.reshape(-1)
        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]
        if len(good_prev) < 8 or len(good_curr) < 8:
            return 0.0, 0.0, 0.0

        m, _ = cv2.estimateAffinePartial2D(
            good_prev,
            good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        if m is None:
            return 0.0, 0.0, 0.0

        dx = float(m[0, 2] / scale)
        dy = float(m[1, 2] / scale)
        da = float(np.arctan2(m[1, 0], m[0, 0]))
        return dx, dy, da

    def update_trajectory(self, dx: float, dy: float, da: float) -> tuple[float, float, float]:
        alpha = self.active_params["alpha"]
        self.trajectory += np.array([dx, dy, da], dtype=np.float32)
        self.smooth_trajectory = alpha * self.smooth_trajectory + (1.0 - alpha) * self.trajectory
        diff = self.smooth_trajectory - self.trajectory
        return dx + float(diff[0]), dy + float(diff[1]), da + float(diff[2])

    def choose_mode(self, dx: float, dy: float, da: float) -> str:
        candidate, motion_state = estimate_motion_state(dx, dy, da, self.mode_history)
        self.last_motion_state = motion_state

        if candidate == self.mode:
            self.pending_mode = None
            self.pending_count = 0
            return self.mode

        required = 3
        if self.mode == "violent" and candidate != "violent":
            required = 5
        elif self.mode == "large_sway" and candidate == "normal":
            required = 4

        if candidate == self.pending_mode:
            self.pending_count += 1
        else:
            self.pending_mode = candidate
            self.pending_count = 1

        if self.pending_count >= required:
            self.mode = candidate
            self.pending_mode = None
            self.pending_count = 0

        return self.mode

    def update(self, frame: np.ndarray) -> np.ndarray:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return frame

        raw_dx, raw_dy, raw_da = self.estimate_transform(self.prev_gray, curr_gray)
        update_mode_history(self.mode_history, raw_dx, raw_dy, raw_da)
        self.mode = self.choose_mode(raw_dx, raw_dy, raw_da)
        self.active_params = self._resolve_mode_params(self.mode)
        dx, dy, da = self.update_trajectory(raw_dx, raw_dy, raw_da)
        dx, dy, da = apply_gain_schedule(dx, dy, da, self.active_params)

        cos_a = np.cos(da)
        sin_a = np.sin(da)
        m = np.array([[cos_a, -sin_a, dx], [sin_a, cos_a, dy]], dtype=np.float32)

        h, w = frame.shape[:2]
        pad = int(self.pad)
        padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101)
        m_pad = m.copy()
        m_pad[0, 2] += pad
        m_pad[1, 2] += pad
        warped = cv2.warpAffine(
            padded,
            m_pad,
            (w + 2 * pad, h + 2 * pad),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        self.prev_gray = curr_gray
        return warped[pad : pad + h, pad : pad + w]
