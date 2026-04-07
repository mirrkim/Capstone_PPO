import numpy as np
import collections

# =============================================
# --- 환경 설정 상수 ---
# =============================================
MAP_SIZE           = 700
START_POS          = np.array([30.0, 30.0])
MIN_OBS_COUNT      = 4
MAX_OBS_COUNT      = 6
OBS_RADIUS_MIN     = 50.0
OBS_RADIUS_MAX     = 100.0
BPSK_SIGNAL_RADIUS  = 250
QAM_SIGNAL_RADIUS   = 150
TARGET_RSSI_THRESHOLD = 0.85
MAX_STEPS = 500
TARGET_MODE = 'random'

BELIEF_DECAY        = 0.95
BELIEF_FREEZE_STEPS = 15

# QAM 위치는 고정이므로 상수로 미리 계산
_QAM_POS = np.array([[400.0, 200.0], [200.0, 400.0]])  # (2, 2)


class DroneEnv:
    def __init__(self):
        self.map_size    = MAP_SIZE
        self.drone_start = START_POS.copy()
        self.state_dim   = 30
        self.action_dim  = 2
        self.MACRO_N     = 2
        self.qam_pos     = [np.array([400, 200]), np.array([200, 400])]

        self.macro_centers = np.array([
            [175.0, 175.0], [525.0, 175.0],
            [175.0, 525.0], [525.0, 525.0]
        ])  # (4, 2) — 벡터 연산용 배열로 변환

        # 장애물 캐시 (reset마다 갱신)
        self._valid_obs = np.empty((0, 2), dtype=np.float32)
        self._valid_r   = np.empty((0,),   dtype=np.float32)

    # ──────────────────────────────────────────────────────────────
    # 장애물 벡터 헬퍼 (모든 루프를 여기서 처리)
    # ──────────────────────────────────────────────────────────────

    def _rebuild_obs_cache(self):
        """_generate_world 후 한 번만 호출. 유효 장애물만 배열로 캐싱."""
        obs_arr = np.array(self.obstacles, dtype=np.float32)   # (N, 2)
        r_arr   = np.array(self.obs_radii,  dtype=np.float32)  # (N,)
        valid   = r_arr > 0
        self._valid_obs = obs_arr[valid]  # (M, 2)
        self._valid_r   = r_arr[valid]    # (M,)

    def _min_dist_to_obs(self, pos):
        """pos 에서 가장 가까운 장애물 표면까지 거리 (없으면 999)."""
        if len(self._valid_obs) == 0:
            return 999.0
        dists = np.linalg.norm(pos - self._valid_obs, axis=1) - self._valid_r
        return float(dists.min())

    def _path_blocked(self, A, B, margin=20.0):
        """A→B 선분이 장애물에 막히는지 벡터화 검사."""
        if len(self._valid_obs) == 0:
            return False
        AB     = B - A
        len_sq = float(np.dot(AB, AB))
        if len_sq < 0.001:
            return False
        # 각 장애물 중심의 선분 위 최근접점 거리
        t       = np.clip(((self._valid_obs - A) @ AB) / len_sq, 0.0, 1.0)  # (M,)
        closest = A + t[:, None] * AB                                         # (M, 2)
        dists   = np.linalg.norm(self._valid_obs - closest, axis=1)          # (M,)
        return bool(np.any(dists < self._valid_r + margin))

    def _path_danger_cost(self, A, B, margin=15.0):
        """A→B 경로가 장애물에 얼마나 가까운지 위험 비용 합산."""
        if len(self._valid_obs) == 0:
            return 0.0
        AB     = B - A
        len_sq = float(np.dot(AB, AB))
        if len_sq < 0.001:
            return 0.0
        t       = np.clip(((self._valid_obs - A) @ AB) / len_sq, 0.0, 1.0)
        closest = A + t[:, None] * AB
        dists   = np.linalg.norm(self._valid_obs - closest, axis=1)
        hit     = dists < (self._valid_r + margin)
        return float(self._valid_r[hit].sum())

    # ──────────────────────────────────────────────────────────────

    def reset(self):
        self.drone_pos   = self.drone_start.copy()
        self.prev_action = np.array([1.0, 1.0]) / np.sqrt(2)
        self.search_vec  = self.prev_action.copy()
        self.steps       = 0
        self.prev_bpsk_rssi = 0.0
        self.belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        self.prev_entropy = -np.sum(self.belief * np.log(self.belief + 1e-8))
        self.current_macro = -1
        self.macro_step_count = 0
        self.macro_entry_freeze = 0
        self.visited_quad_history = collections.deque(maxlen=4)
        self.chosen_target = -1

        self._generate_world()
        return self.get_state()

    def _generate_world(self):
        self.obstacles = []; self.obs_radii = []
        num_obs = np.random.randint(MIN_OBS_COUNT, MAX_OBS_COUNT + 1)
        for _ in range(num_obs):
            r  = np.random.uniform(OBS_RADIUS_MIN, OBS_RADIUS_MAX)
            px = np.random.uniform(r + 10, self.map_size - r - 10)
            py = np.random.uniform(r + 10, self.map_size - r - 10)
            if px < 180 and py < 180:
                continue
            self.obstacles.append(np.array([px, py]))
            self.obs_radii.append(r)

        while len(self.obstacles) < MAX_OBS_COUNT:
            self.obstacles.append(np.array([-1000.0, -1000.0]))
            self.obs_radii.append(0.0)

        while True:
            px, py = np.random.uniform(50, 650), np.random.uniform(50, 650)
            if px + py > 700:
                candidate_pos = np.array([px, py])
                self._rebuild_obs_cache()  # 미리 캐시 구성해서 루프 없이 검사
                if len(self._valid_obs) == 0 or \
                   np.min(np.linalg.norm(candidate_pos - self._valid_obs, axis=1) - self._valid_r) >= 30.0:
                    self.bpsk_pos = candidate_pos
                    break

    def get_bpsk_rssi(self, pos):
        dist = np.linalg.norm(self.bpsk_pos - pos)
        return max(0.0, 1.0 - dist / BPSK_SIGNAL_RADIUS) if dist < BPSK_SIGNAL_RADIUS else 0.0

    def get_state(self):
        curr_bpsk = self.get_bpsk_rssi(self.drone_pos)
        if curr_bpsk > 0.0:
            diff = self.bpsk_pos - self.drone_pos
            sensor_dir = diff / (np.linalg.norm(diff) + 1e-8)
        else:
            sensor_dir = self.search_vec

        # QAM AoA — (2, 2) 배열 연산으로 루프 제거
        qam_diffs = _QAM_POS - self.drone_pos          # (2, 2)
        qam_dists = np.linalg.norm(qam_diffs, axis=1)  # (2,)
        in_range  = qam_dists < QAM_SIGNAL_RADIUS
        qam_dirs  = np.where(
            in_range[:, None],
            qam_diffs / (qam_dists[:, None] + 1e-8),
            np.zeros((2, 2))
        )  # (2, 2)
        qam_aoas = qam_dirs.ravel()  # [dx0, dy0, dx1, dy1]

        state = [
            self.drone_pos[0] / 700, self.drone_pos[1] / 700,
            (700 - self.drone_pos[0]) / 700, (700 - self.drone_pos[1]) / 700,
            curr_bpsk, (curr_bpsk - self.prev_bpsk_rssi) * 100,
            self.prev_action[0], self.prev_action[1],
            sensor_dir[0], sensor_dir[1],
            qam_aoas[0], qam_aoas[1], qam_aoas[2], qam_aoas[3],
            self.belief[0], self.belief[1], self.belief[2], self.belief[3],
        ]

        # 장애물 정보 — 벡터화 (루프 없음)
        if len(self._valid_obs) > 0:
            rel_pos  = (self._valid_obs - self.drone_pos) / 700          # (M, 2)
            dists    = np.linalg.norm(rel_pos, axis=1) - self._valid_r / 700  # (M,)
            sort_idx = np.argsort(dists)[:4]
            for idx in sort_idx:
                state.extend([rel_pos[idx, 0], rel_pos[idx, 1], max(0.0, float(dists[idx]))])
            for _ in range(4 - len(sort_idx)):
                state.extend([0.0, 0.0, 0.0])
        else:
            state.extend([0.0] * 12)

        return np.array(state, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # ── 진행 방향 부드럽게 혼합 ─────────────────────────────────
        pre_min_dist  = self._min_dist_to_obs(self.drone_pos)
        pre_wall_dist = float(min(self.drone_pos[0], self.drone_pos[1],
                                  700 - self.drone_pos[0], 700 - self.drone_pos[1]))
        danger_level = max(1.0 - np.clip(pre_min_dist / 35.0, 0.0, 1.0),
                           1.0 - np.clip(pre_wall_dist / 40.0, 0.0, 1.0))
        alpha    = 0.40 - 0.15 * danger_level
        velocity = action * alpha + self.prev_action * (1.0 - alpha)
        v_norm   = np.linalg.norm(velocity)
        velocity = velocity / v_norm if v_norm > 0.001 else self.prev_action

        self.prev_action = velocity
        new_pos = self.drone_pos + velocity * 10.0

        # ── 충돌 검사 ───────────────────────────────────────────────
        if new_pos[0] < 0 or new_pos[0] > 700 or new_pos[1] < 0 or new_pos[1] > 700:
            return self.get_state(), -2000.0, True, "Wall Crash"

        if len(self._valid_obs) > 0:
            if np.any(np.linalg.norm(new_pos - self._valid_obs, axis=1) <= self._valid_r):
                return self.get_state(), -3000.0, True, "Obs Crash"

        # ── 소프트 패널티 (근접 경고) ───────────────────────────────
        obs_margin       = 70.0
        wall_margin      = 70.0
        near_crash_margin = 20.0

        min_dist_to_obs = self._min_dist_to_obs(new_pos)  # 한 번만 계산

        soft_penalty = 0.0
        if min_dist_to_obs < obs_margin:
            soft_penalty -= ((obs_margin - min_dist_to_obs) / obs_margin) ** 2 * 150.0
        if min_dist_to_obs < near_crash_margin:
            soft_penalty -= 300.0

        wall_dist = float(min(new_pos[0], new_pos[1], 700 - new_pos[0], 700 - new_pos[1]))
        if wall_dist < wall_margin:
            soft_penalty -= ((wall_margin - wall_dist) / wall_margin) ** 2 * 130.0

        # ── 위치·신호 업데이트 ──────────────────────────────────────
        self.drone_pos = new_pos
        self.steps    += 1
        curr_bpsk  = self.get_bpsk_rssi(new_pos)
        bpsk_diff  = curr_bpsk - self.prev_bpsk_rssi

        # ── 매크로 쿼드런트 업데이트 ────────────────────────────────
        quad_idx = int(np.clip(new_pos[1] / 350.0, 0, 1)) * 2 + \
                   int(np.clip(new_pos[0] / 350.0, 0, 1))
        macro_crossing_bonus = 0.0
        if quad_idx != self.current_macro:
            if quad_idx not in self.visited_quad_history:
                macro_crossing_bonus = 300.0
            self.visited_quad_history.append(self.current_macro)
            self.current_macro      = quad_idx
            self.macro_step_count   = 0
            self.macro_entry_freeze = BELIEF_FREEZE_STEPS
        else:
            self.macro_step_count += 1
            if self.macro_entry_freeze > 0:
                self.macro_entry_freeze -= 1

        # ── Belief 업데이트 ─────────────────────────────────────────
        if curr_bpsk > 0.0:
            self.belief[:] = 0.0
            self.belief[quad_idx] = 1.0
        else:
            if self.macro_entry_freeze == 0:
                self.belief[quad_idx] *= BELIEF_DECAY
                self.belief /= np.sum(self.belief)

        curr_entropy  = -float(np.sum(self.belief * np.log(self.belief + 1e-8)))
        info_reward   = (self.prev_entropy - curr_entropy) * 10.0
        self.prev_entropy = curr_entropy

        # ── 목표 도달 ───────────────────────────────────────────────
        if curr_bpsk >= TARGET_RSSI_THRESHOLD:
            return self.get_state(), 5000.0, True, "Target Found"

        # ── 보상 계산 ───────────────────────────────────────────────
        step_penalty = -0.8

        if curr_bpsk > 0.0:
            rssi_diff_rew    = bpsk_diff * 300.0
            proximity_reward = curr_bpsk * 80.0
            if self._path_blocked(new_pos, self.bpsk_pos, margin=20.0):
                proximity_reward *= 0.2
            reward = step_penalty + rssi_diff_rew + proximity_reward + soft_penalty + info_reward

        else:
            self.search_vec = velocity.copy()

            # 현재 쿼드런트 제외한 belief 최고 후보 선택
            temp_belief = self.belief.copy()
            temp_belief[quad_idx] = -1.0
            max_p      = float(np.max(temp_belief))
            candidates = np.where(np.isclose(temp_belief, max_p))[0]

            # Sticky Target 로직
            if self.chosen_target in candidates:
                best_target = self.chosen_target
            else:
                if len(candidates) > 1:
                    # 각 후보까지 거리 + 경로 위험 비용 벡터화
                    target_centers = self.macro_centers[candidates]       # (K, 2)
                    dist_costs     = np.linalg.norm(
                        target_centers - self.drone_pos, axis=1)          # (K,)
                    danger_costs   = np.array([
                        self._path_danger_cost(self.drone_pos, tc)
                        for tc in target_centers
                    ])                                                    # (K,) — K≤4
                    total_costs    = dist_costs + danger_costs * 3.0
                    best_target    = int(candidates[np.argmin(total_costs)])
                else:
                    best_target = int(candidates[0])

                self.chosen_target = best_target

            target_pos     = self.macro_centers[best_target]
            dir_to_target  = target_pos - self.drone_pos
            dir_to_target /= np.linalg.norm(dir_to_target) + 1e-8
            danger_factor  = float(np.clip(min_dist_to_obs / obs_margin, 0.0, 1.0))

            belief_diff = float(np.clip(
                self.belief[best_target] - self.belief[quad_idx], 0.0, 0.5))
            multiplier  = 30.0 + belief_diff * 180.0
            drive_reward = float(np.dot(velocity, dir_to_target)) * multiplier * danger_factor

            reward = step_penalty + drive_reward + macro_crossing_bonus + info_reward + soft_penalty

        self.prev_bpsk_rssi = curr_bpsk
        return self.get_state(), reward, self.steps >= MAX_STEPS, "Normal"