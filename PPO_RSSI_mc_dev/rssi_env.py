import math
import numpy as np
import pandas as pd

from kalman_filter import kalman_filter


# ============================================================
# Basic settings
# ============================================================
K_SAMPLES = 20
PARTIAL_NLOS_RADIUS = 12.0

OBSTACLES = [
    (15, 20, 35, 40),
    (55, 15, 75, 38),
    (20, 58, 45, 72),
    (60, 55, 82, 75),
]


# ============================================================
# Geometry
# ============================================================
def _seg_intersect_rect(p1, p2, rect):
    x1, y1 = p1
    x2, y2 = p2
    xmn, ymn, xmx, ymx = rect

    def code(x, y):
        c = 0
        if x < xmn:
            c |= 1
        elif x > xmx:
            c |= 2

        if y < ymn:
            c |= 4
        elif y > ymx:
            c |= 8

        return c

    c1, c2 = code(x1, y1), code(x2, y2)

    while True:
        if not (c1 | c2):
            return True
        if c1 & c2:
            return False

        c = c1 if c1 else c2

        if c & 1:
            x = xmn
            y = y1 + (y2 - y1) * (xmn - x1) / (x2 - x1 + 1e-12)
        elif c & 2:
            x = xmx
            y = y1 + (y2 - y1) * (xmx - x1) / (x2 - x1 + 1e-12)
        elif c & 4:
            x = x1 + (x2 - x1) * (ymn - y1) / (y2 - y1 + 1e-12)
            y = ymn
        else:
            x = x1 + (x2 - x1) * (ymx - y1) / (y2 - y1 + 1e-12)
            y = ymx

        if c == c1:
            x1, y1, c1 = x, y, code(x, y)
        else:
            x2, y2, c2 = x, y, code(x, y)


def is_nlos(drone_pos, target_pos):
    """
    Full NLOS if the drone-target straight path intersects an obstacle.
    """
    return any(_seg_intersect_rect(drone_pos, target_pos, obs) for obs in OBSTACLES)


def _in_obstacle(pos, margin=0.5):
    x, y = float(pos[0]), float(pos[1])
    return any(
        o[0] - margin <= x <= o[2] + margin and
        o[1] - margin <= y <= o[3] + margin
        for o in OBSTACLES
    )


def rot2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def circle_xy(center, radius):
    t = np.linspace(0, 2 * np.pi, 360)
    return center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)


def make_one_scenario(rng, area_min=10.0, area_max=90.0, r_form=None, offset=5.0):
    """
    Create one random enemy position and three drone positions.
    """
    if r_form is None:
        r_form = float(rng.uniform(3, 20))  # 0.7 + (95,120,145)px formation 커버
    for _ in range(500):
        x_true = rng.uniform(area_min, area_max, 2)

        if _in_obstacle(x_true):
            continue

        centroid = x_true + rng.uniform(-offset, offset, 2)
        angles = np.array([0.0, 2 * math.pi / 3, 4 * math.pi / 3])
        R = rot2(rng.uniform(0, 2 * math.pi))

        A = np.array([
            R @ np.array([r_form * math.cos(a), r_form * math.sin(a)]) + centroid
            for a in angles
        ])

        if any(_in_obstacle(A[k]) for k in range(3)):
            continue

        return x_true, A

    x_true = rng.uniform(40, 60, 2)
    R = rot2(rng.uniform(0, 2 * math.pi))

    A = np.array([
        R @ np.array([r_form * math.cos(a), r_form * math.sin(a)]) + x_true
        for a in [0.0, 2 * math.pi / 3, 4 * math.pi / 3]
    ])

    return x_true, A


# ============================================================
# RSSI channel
# ============================================================
class RealDataChannel:
    """
    RSSI channel using measured Excel datasets.

    link_type:
    0 = LOS
    1 = Partial NLOS
    2 = Full NLOS
    """

    def __init__(self, no_obs_path, with_obs_path, rng):
        self.rng = rng

        def load(path):
            df = pd.read_excel(path, engine="openpyxl", header=2)
            df = df.sort_values("distance_m").reset_index(drop=True)

            dists = df["distance_m"].values.astype(float)
            sample_cols = [c for c in df.columns if str(c).startswith("sample_")]

            pools = {}
            for _, row in df.iterrows():
                d = row["distance_m"]
                pools[d] = row[sample_cols].values.astype(float)

            return dists, pools

        self.los_d, self.los_pools = load(no_obs_path)
        self.nlos_d, self.nlos_pools = load(with_obs_path)

    def _nearest_bin(self, d, dist_arr):
        idx = np.argmin(np.abs(dist_arr - d))
        return dist_arr[idx]

    def _nearest_obs_dist(self, pos):
        x, y = float(pos[0]), float(pos[1])
        min_d = np.inf

        for xmn, ymn, xmx, ymx in OBSTACLES:
            cx = np.clip(x, xmn, xmx)
            cy = np.clip(y, ymn, ymx)
            min_d = min(min_d, math.sqrt((x - cx) ** 2 + (y - cy) ** 2))

        return min_d

    def _draw(self, d, pools, dist_arr, K):
        bin_d = self._nearest_bin(d, dist_arr)
        pool = pools[bin_d]
        return self.rng.choice(pool, size=K, replace=True)

    def sample(self, drone_pos, target_pos, K=K_SAMPLES, near_r=PARTIAL_NLOS_RADIUS):
        """
        Sample K RSSI values and return:
        raw_mean, kalman_mean, raw_std, link_type

        link_type:
          0 = LOS   (장애물 없음)
          1 = PNLOS (드론 근처 장애물)
          2 = NLOS  (경로 차단)
        """
        d = float(np.linalg.norm(np.array(drone_pos) - np.array(target_pos)))

        # Full NLOS: 경로 차단
        if is_nlos(drone_pos, target_pos):
            raw = self._draw(max(d, 5.0), self.nlos_pools, self.nlos_d, K)
            return raw.mean(), kalman_filter(raw), raw.std(), 2

        # Partial NLOS: 드론 근처 장애물 → LOS 10개 + NLOS 10개 고정
        obs_d = self._nearest_obs_dist(drone_pos)
        if obs_d < near_r:
            half = K // 2
            raw_los  = self.rng.choice(
                self._draw(d, self.los_pools, self.los_d, K), half, replace=True)
            raw_nlos = self.rng.choice(
                self._draw(max(d, 5.0), self.nlos_pools, self.nlos_d, K), half, replace=True)
            raw = np.concatenate([raw_los, raw_nlos])
            self.rng.shuffle(raw)
            return raw.mean(), kalman_filter(raw), raw.std(), 1

        # LOS
        raw = self._draw(d, self.los_pools, self.los_d, K)
        return raw.mean(), kalman_filter(raw), raw.std(), 0


# ============================================================
# Distance estimation and trilateration
# ============================================================
def dist_from_rssi_baseline(rssi_mean, A0=-40.85, n=2.201):
    return np.power(10.0, (A0 - rssi_mean) / (10.0 * n))


def trilat_2d_ls(A, d):
    x1, y1 = A[0]
    x2, y2 = A[1]
    x3, y3 = A[2]

    d1, d2, d3 = float(d[0]), float(d[1]), float(d[2])

    M = np.array([
        [2 * (x2 - x1), 2 * (y2 - y1)],
        [2 * (x3 - x1), 2 * (y3 - y1)],
    ])

    b = np.array([
        d1**2 - d2**2 + x2**2 - x1**2 + y2**2 - y1**2,
        d1**2 - d3**2 + x3**2 - x1**2 + y3**2 - y1**2,
    ])

    x_hat, *_ = np.linalg.lstsq(M, b, rcond=None)
    return x_hat.astype(float)
