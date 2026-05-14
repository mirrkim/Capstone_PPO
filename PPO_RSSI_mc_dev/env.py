import numpy as np
import collections
import scipy.ndimage # 🚀 SDF(거리 맵) 생성을 위한 필수 라이브러리

# =============================================
# --- 환경 설정 상수 ---
# =============================================
MAP_SIZE           = 700
START_POS          = np.array([30.0, 30.0])
MIN_OBS_COUNT      = 4
MAX_OBS_COUNT      = 8    # 모양이 섞이므로 갯수를 조금 넉넉히
OBS_RADIUS_MIN     = 40.0 
OBS_RADIUS_MAX     = 90.0 
BPSK_SIGNAL_RADIUS  = 250
QAM_SIGNAL_RADIUS   = 150
TARGET_RSSI_THRESHOLD = 0.85
MAX_STEPS = 512
TARGET_MODE = 'random'

BELIEF_DECAY        = 0.95
BELIEF_FREEZE_STEPS = int(MAP_SIZE * 0.01)   

_QAM_POS = np.array([[400.0, 200.0], [200.0, 400.0]])

# ── 라이다(LiDAR) 설정 ─────────────────────────────────────────────
NUM_LIDAR_RAYS  = 16      
LIDAR_MAX_RANGE = 200.0   


class DroneEnv:
    def __init__(self):
        self.map_size    = MAP_SIZE
        self.drone_start = START_POS.copy()
        
        self.MACRO_N     = 3
        self.macro_count = self.MACRO_N * self.MACRO_N
        self.state_dim   = 14 + self.macro_count + NUM_LIDAR_RAYS
        self.action_dim  = 2
        self.qam_pos     = [np.array([400, 200]), np.array([200, 400])]

        step_size = self.map_size / self.MACRO_N
        self.macro_centers = np.array([
            [step_size * x + step_size / 2.0, step_size * y + step_size / 2.0]
            for y in range(self.MACRO_N) for x in range(self.MACRO_N)
        ])

        # 라이다 방향벡터 사전 계산
        _angles = np.linspace(0.0, 2.0 * np.pi, NUM_LIDAR_RAYS, endpoint=False)
        self._lidar_dirs = np.stack(
            [np.cos(_angles), np.sin(_angles)], axis=1
        ).astype(np.float32)

    def _generate_world(self):
        """
        [격자 기반 맵 생성] 
        어떤 모양이든 픽셀(격자)에 색칠하듯 그려넣고, SDF(거리장)를 추출합니다.
        """
        # 1. 빈 캔버스(격자) 준비 (False: 빈 공간, True: 장애물)
        self.occupancy = np.zeros((int(self.map_size), int(self.map_size)), dtype=bool)

        # 맵 외곽선을 장애물로 처리하여 라이다가 맵 밖을 벽으로 인식하게 함
        self.occupancy[0:2, :] = True
        self.occupancy[-2:, :] = True
        self.occupancy[:, 0:2] = True
        self.occupancy[:, -2:] = True

        # 2. 다양한 형태의 장애물을 캔버스에 새기기 (도장 찍기)
        num_obs = np.random.randint(MIN_OBS_COUNT, MAX_OBS_COUNT + 1)
        y_grid, x_grid = np.ogrid[0:int(self.map_size), 0:int(self.map_size)]

        for _ in range(num_obs):
            obs_type = np.random.choice(['circle', 'rect'])
            r = np.random.uniform(OBS_RADIUS_MIN, OBS_RADIUS_MAX)
            px = np.random.uniform(r + 10, self.map_size - r - 10)
            py = np.random.uniform(r + 10, self.map_size - r - 10)

            if px < 180 and py < 180: continue # 시작 지점 보호

            if obs_type == 'circle':
                # 원형 격자 마스크 생성
                mask = (x_grid - px)**2 + (y_grid - py)**2 <= r**2
                self.occupancy[mask] = True
            else:
                # 직사각형(건물) 격자 마스크 생성 (가로, 세로 비율 랜덤)
                half_w = r * np.random.uniform(0.6, 1.2)
                half_h = r * np.random.uniform(0.6, 1.2)
                x_min, x_max = int(max(0, px - half_w)), int(min(self.map_size, px + half_w))
                y_min, y_max = int(max(0, py - half_h)), int(min(self.map_size, py + half_h))
                self.occupancy[y_min:y_max, x_min:x_max] = True

        # 🚀 3. SDF (Signed Distance Field) 맵 생성! (연산 속도의 핵심)
        # distance_transform_edt는 False(0)인 곳으로부터의 거리를 계산하므로
        # 장애물(True)을 거꾸로 False(~occupancy)로 반전시켜 계산합니다.
        empty_space = ~self.occupancy
        self.sdf_map = scipy.ndimage.distance_transform_edt(empty_space).astype(np.float32)

        # 4. 안전한 곳에 BPSK 타겟 배치 (SDF 맵 활용)
        while True:
            px, py = np.random.uniform(50, self.map_size - 50), np.random.uniform(50, self.map_size - 50)
            if px + py > self.map_size:
                cx, cy = int(px), int(py)
                # SDF 맵을 확인하여 장애물로부터 30픽셀 이상 떨어진 안전지대인지 확인
                if self.sdf_map[cy, cx] >= 30.0:
                    self.bpsk_pos = np.array([px, py])
                    break

    def _min_dist_to_obs(self, pos):
        """[수정] SDF 맵에서 좌표값만 읽어오면 끝! 연산량 O(1)"""
        cx = int(np.clip(pos[0], 0, self.map_size - 1))
        cy = int(np.clip(pos[1], 0, self.map_size - 1))
        return float(self.sdf_map[cy, cx])

    def _cast_lidar_rays(self, pos):
        """
        [수정] Sphere Tracing (구면 추적) 기법을 활용한 초고속 라이다.
        조금씩 전진하는 게 아니라, SDF가 알려준 안전거리만큼 한 번에 '점프'합니다.
        """
        t = np.zeros(NUM_LIDAR_RAYS, dtype=np.float32)

        # 그래픽스 렌더링에서 쓰는 Ray Marching 최대 반복 횟수
        for _ in range(30):
            curr_pos = pos + t[:, None] * self._lidar_dirs
            cx = np.clip(curr_pos[:, 0], 0, self.map_size - 1).astype(int)
            cy = np.clip(curr_pos[:, 1], 0, self.map_size - 1).astype(int)

            # 현재 광선 끝점들에서의 장애물까지의 거리
            dists = self.sdf_map[cy, cx]
            
            # 안전거리(dists)만큼 광선을 한 번에 연장
            t += dists

            # 모든 광선이 벽에 박았거나(거리 1픽셀 미만), 최대 거리에 도달하면 루프 종료
            if np.all((dists < 1.0) | (t >= LIDAR_MAX_RANGE)):
                break

        return np.clip(t / LIDAR_MAX_RANGE, 0.0, 1.0)

    def _path_blocked(self, A, B, margin=20.0):
        """[수정] 선분 AB를 일정 간격으로 쪼개서 SDF 맵을 스캔"""
        dist_AB = np.linalg.norm(B - A)
        if dist_AB < 1.0: return False
        
        steps = max(2, int(dist_AB / 5.0)) # 5픽셀 단위로 샘플링
        t_vals = np.linspace(0, 1, steps)
        pts = A + t_vals[:, None] * (B - A)

        cx = np.clip(pts[:, 0], 0, self.map_size - 1).astype(int)
        cy = np.clip(pts[:, 1], 0, self.map_size - 1).astype(int)
        
        min_d = np.min(self.sdf_map[cy, cx])
        return bool(min_d < margin)

    def _path_danger_cost(self, A, B, margin=15.0):
        """[수정] 경로 상에 margin보다 가까운 위험 요소(SDF 값)들의 합산 비용"""
        dist_AB = np.linalg.norm(B - A)
        if dist_AB < 1.0: return 0.0
        
        steps = max(2, int(dist_AB / 5.0))
        t_vals = np.linspace(0, 1, steps)
        pts = A + t_vals[:, None] * (B - A)

        cx = np.clip(pts[:, 0], 0, self.map_size - 1).astype(int)
        cy = np.clip(pts[:, 1], 0, self.map_size - 1).astype(int)
        
        dists = self.sdf_map[cy, cx]
        hit = dists < margin
        return float(np.sum(margin - dists[hit]))

    def _calc_soft_penalty(self, pos, min_dist_obs, obs_margin=50.0, wall_margin=30.0, near_crash_margin=20.0, in_signal=False):
        scale = 2.0 if in_signal else 1.0
        soft = 0.0
        if min_dist_obs < obs_margin:
            soft -= ((obs_margin - min_dist_obs) / obs_margin) ** 2 * 150.0 * scale
        if min_dist_obs < near_crash_margin:
            soft -= 300.0 * scale

        return soft

    def reset(self):
        self.drone_pos   = self.drone_start.copy()
        self.prev_action = np.array([1.0, 1.0]) / np.sqrt(2)
        self.search_vec  = self.prev_action.copy()
        self.steps       = 0
        self.prev_bpsk_rssi = 0.0
        self.belief = np.full(self.macro_count, 1.0 / self.macro_count, dtype=np.float32)
        self.area_visit_steps = np.zeros(self.macro_count, dtype=int)
        self.current_macro = -1
        self.macro_entry_freeze = 0
        self.visited_quad_history = collections.deque(maxlen=self.macro_count)
        self.chosen_target = -1
        self._generate_world()
        return self.get_state()

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

        qam_diffs = _QAM_POS - self.drone_pos
        qam_dists = np.linalg.norm(qam_diffs, axis=1)
        in_range  = qam_dists < QAM_SIGNAL_RADIUS
        qam_dirs  = np.where(
            in_range[:, None],
            qam_diffs / (qam_dists[:, None] + 1e-8),
            np.zeros((2, 2))
        )
        qam_aoas = qam_dirs.ravel()

        state = [
            self.drone_pos[0] / self.map_size, self.drone_pos[1] / self.map_size,
            (self.map_size - self.drone_pos[0]) / self.map_size, (self.map_size - self.drone_pos[1]) / self.map_size,
            curr_bpsk, (curr_bpsk - self.prev_bpsk_rssi) * 100,
            self.prev_action[0], self.prev_action[1],
            sensor_dir[0], sensor_dir[1],
            qam_aoas[0], qam_aoas[1], qam_aoas[2], qam_aoas[3],
        ]
        
        state.extend(self.belief)
        state.extend(self._cast_lidar_rays(self.drone_pos))

        return np.array(state, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        pre_min_dist  = self._min_dist_to_obs(self.drone_pos)
        pre_wall_dist = float(min(self.drone_pos[0], self.drone_pos[1],
                                  self.map_size - self.drone_pos[0], self.map_size - self.drone_pos[1]))
        danger_level = max(1.0 - np.clip(pre_min_dist / 50.0, 0.0, 1.0),
                           1.0 - np.clip(pre_wall_dist / 50.0, 0.0, 1.0))
        alpha    = 0.40 + 0.40 * danger_level
        velocity = action * alpha + self.prev_action * (1.0 - alpha)
        v_norm   = np.linalg.norm(velocity)
        velocity = velocity / v_norm if v_norm > 0.001 else self.prev_action
        self.prev_action = velocity
        new_pos = self.drone_pos + velocity * 8.0

        if new_pos[0] < 0 or new_pos[0] > self.map_size or new_pos[1] < 0 or new_pos[1] > self.map_size:
            return self.get_state(), -2000.0, True, "Wall Crash"
        
        # 🚀 [수정] 충돌 판정도 SDF로 처리 (가장 가까운 장애물 거리가 0 근처면 충돌)
        if self._min_dist_to_obs(new_pos) < 2.0:
            return self.get_state(), -3000.0, True, "Obs Crash"

        self.drone_pos = new_pos
        self.steps    += 1
        curr_bpsk  = self.get_bpsk_rssi(new_pos)
        bpsk_diff  = curr_bpsk - self.prev_bpsk_rssi
        min_dist_to_obs = self._min_dist_to_obs(new_pos)
        soft_penalty = self._calc_soft_penalty(new_pos, min_dist_to_obs, in_signal=(curr_bpsk > 0.0))

        grid_step = self.map_size / self.MACRO_N
        idx_x = int(np.clip(new_pos[0] / grid_step, 0, self.MACRO_N - 1))
        idx_y = int(np.clip(new_pos[1] / grid_step, 0, self.MACRO_N - 1))
        quad_idx = idx_y * self.MACRO_N + idx_x

        macro_crossing_bonus = 0.0
        if quad_idx != self.current_macro:
            if quad_idx not in self.visited_quad_history:
                macro_crossing_bonus = 250.0
            self.current_macro = quad_idx
            self.chosen_target = -1 
            self.visited_quad_history.append(quad_idx)

        self.area_visit_steps[quad_idx] += 1
        steps_here = self.area_visit_steps[quad_idx]

        if steps_here <= BELIEF_FREEZE_STEPS:
            self.macro_entry_freeze = 1
        else:
            self.macro_entry_freeze = 0

        if curr_bpsk > 0.0:
            self.belief[:] = 0.0
            self.belief[quad_idx] = 1.0
        else:
            if self.macro_entry_freeze == 0:
                over_steps = steps_here - BELIEF_FREEZE_STEPS
                dynamic_decay = max(0.80, 0.96 - (over_steps * 0.008)) 
                self.belief[quad_idx] *= dynamic_decay
                self.belief /= np.sum(self.belief)

        if curr_bpsk >= TARGET_RSSI_THRESHOLD:
            return self.get_state(), 5000.0, True, "Target Found"

        step_penalty = -3

        if curr_bpsk > 0.0: 
            safety_factor = float(np.clip(min_dist_to_obs / 70.0, 0.0, 1.0))
            rssi_diff_rew    = bpsk_diff * 300.0
            proximity_reward = curr_bpsk * 80.0
            if self._path_blocked(new_pos, self.bpsk_pos, margin=20.0):
                proximity_reward *= 0.2
                rssi_diff_rew    *= 0.3
            signal_reward = (rssi_diff_rew + proximity_reward) * safety_factor
            reward = step_penalty + signal_reward + soft_penalty + macro_crossing_bonus
        else:
            self.search_vec = velocity.copy()
            temp_belief = self.belief.copy()
            temp_belief[quad_idx] = -1.0
            max_p      = float(np.max(temp_belief))
            candidates = np.where(np.isclose(temp_belief, max_p))[0]

            if self.chosen_target in candidates:
                best_target = self.chosen_target
            else:
                if len(candidates) > 1:
                    target_centers = self.macro_centers[candidates]
                    dist_costs = np.linalg.norm(target_centers - self.drone_pos, axis=1)
                    danger_costs = np.array([self._path_danger_cost(self.drone_pos, tc) for tc in target_centers])
                    dest_danger = np.array([max(0.0, 80.0 - self._min_dist_to_obs(tc)) for tc in target_centers])
                    total_costs  = dist_costs + danger_costs * 15.0 + dest_danger * 30.0
                    best_target  = int(candidates[np.argmin(total_costs)])
                else:
                    best_target = int(candidates[0])
                self.chosen_target = best_target

            target_pos    = self.macro_centers[best_target]
            dir_to_target = target_pos - self.drone_pos
            dir_to_target /= np.linalg.norm(dir_to_target) + 1e-8
            belief_diff  = float(np.clip(self.belief[best_target] - self.belief[quad_idx], 0.0, 0.5))
            dot_val = float(np.dot(velocity, dir_to_target))
            multiplier = 150.0 + belief_diff * 500.0
            drive_reward = dot_val * multiplier
            reward = step_penalty + drive_reward + soft_penalty + macro_crossing_bonus

        self.prev_bpsk_rssi = curr_bpsk
        return self.get_state(), reward, self.steps >= MAX_STEPS, "Normal"