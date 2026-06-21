import os
# ── TensorFlow / NumPy 로그·경고 억제 (반드시 TF import 보다 먼저) ──
# 0=all, 1=INFO 숨김, 2=WARNING 숨김, 3=ERROR만
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')   # oneDNN 안내 메시지 제거
os.environ.setdefault('TF_CPP_MIN_VLOG_LEVEL', '3')

import warnings
# RML 데이터셋을 pickle.load 할 때 NumPy가 내는 dtype align 폐기예정 경고 억제
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import collections
import scipy.ndimage

# NumPy 버전에 따라 존재하는 전용 경고 클래스도 함께 억제
try:
    warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
except AttributeError:
    pass

# =============================================
# --- 환경 설정 상수 ---
# =============================================
MAP_SIZE           = 700
START_POS          = np.array([30.0, 30.0])
MIN_OBS_COUNT      = 5
MAX_OBS_COUNT      = 7
OBS_RADIUS_MIN     = 40.0 
OBS_RADIUS_MAX     = 90.0 
BPSK_SIGNAL_RADIUS  = 250
QAM_SIGNAL_RADIUS   = 150
TARGET_RSSI_THRESHOLD = 0.85
MAX_STEPS = 512
TARGET_MODE = 'random'

AMC_CLASSIFY_INTERVAL = 5    # 한 번 감지하면 9999스텝 동안 결과 고정!

BELIEF_DECAY        = 0.95
BELIEF_FREEZE_STEPS = int(MAP_SIZE * 0.01)   

_QAM_POS = np.array([[400.0, 200.0], [200.0, 400.0]])

# ── 라이다(LiDAR) 설정 ──
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

        # ── AMC (자동 변조 분류) ──────────────────────────────────
        self.detected_signal_type = 0   # 0=없음  1=BPSK  2=QAM
        self._amc_cache_steps = 0
        self._amc_cache_type  = 0
        self._amc_model  = None
        self._rml_raw    = None
        self._mods       = None
        self._snrs_list  = None
        _has_model = os.path.exists('my_amc_model.h5')
        _has_data  = os.path.exists('RML2016.10a_dict.dat')
        self.use_amc = _has_model and _has_data
        if self.use_amc:
            self._load_amc()
        else:
            print('[AMC] 모델/데이터 없음 → 거리 기반 완벽 분류 사용')

        _angles = np.linspace(0.0, 2.0 * np.pi, NUM_LIDAR_RAYS, endpoint=False)
        self._lidar_dirs = np.stack(
            [np.cos(_angles), np.sin(_angles)], axis=1
        ).astype(np.float32)

    def _generate_world(self):
        self.occupancy = np.zeros((int(self.map_size), int(self.map_size)), dtype=bool)
        self.occupancy[0:2, :] = True
        self.occupancy[-2:, :] = True
        self.occupancy[:, 0:2] = True
        self.occupancy[:, -2:] = True

        num_obs = np.random.randint(MIN_OBS_COUNT, MAX_OBS_COUNT + 1)
        y_grid, x_grid = np.ogrid[0:int(self.map_size), 0:int(self.map_size)]

        for _ in range(num_obs):
            obs_type = np.random.choice(['circle', 'rect'])
            r = np.random.uniform(OBS_RADIUS_MIN, OBS_RADIUS_MAX)
            px = np.random.uniform(r + 10, self.map_size - r - 10)
            py = np.random.uniform(r + 10, self.map_size - r - 10)

            if px < 180 and py < 180: continue 

            if obs_type == 'circle':
                mask = (x_grid - px)**2 + (y_grid - py)**2 <= r**2
                self.occupancy[mask] = True
            else:
                half_w = r * np.random.uniform(0.6, 1.2)
                half_h = r * np.random.uniform(0.6, 1.2)
                x_min, x_max = int(max(0, px - half_w)), int(min(self.map_size, px + half_w))
                y_min, y_max = int(max(0, py - half_h)), int(min(self.map_size, py + half_h))
                self.occupancy[y_min:y_max, x_min:x_max] = True

        empty_space = ~self.occupancy
        self.sdf_map = scipy.ndimage.distance_transform_edt(empty_space).astype(np.float32)

        while True:
            px, py = np.random.uniform(50, self.map_size - 50), np.random.uniform(50, self.map_size - 50)
            if px + py > self.map_size:
                cx, cy = int(px), int(py)
                if self.sdf_map[cy, cx] >= 30.0:
                    self.bpsk_pos = np.array([px, py])
                    break

    def _min_dist_to_obs(self, pos):
        cx = int(np.clip(pos[0], 0, self.map_size - 1))
        cy = int(np.clip(pos[1], 0, self.map_size - 1))
        return float(self.sdf_map[cy, cx])

    def _cast_lidar_rays(self, pos):
        t = np.zeros(NUM_LIDAR_RAYS, dtype=np.float32)
        for _ in range(30):
            curr_pos = pos + t[:, None] * self._lidar_dirs
            cx = np.clip(curr_pos[:, 0], 0, self.map_size - 1).astype(int)
            cy = np.clip(curr_pos[:, 1], 0, self.map_size - 1).astype(int)
            dists = self.sdf_map[cy, cx]
            t += dists
            if np.all((dists < 1.0) | (t >= LIDAR_MAX_RANGE)):
                break
        return np.clip(t / LIDAR_MAX_RANGE, 0.0, 1.0)

    def _path_blocked(self, A, B, margin=20.0):
        dist_AB = np.linalg.norm(B - A)
        if dist_AB < 1.0: return False
        steps = max(2, int(dist_AB / 5.0))
        t_vals = np.linspace(0, 1, steps)
        pts = A + t_vals[:, None] * (B - A)
        cx = np.clip(pts[:, 0], 0, self.map_size - 1).astype(int)
        cy = np.clip(pts[:, 1], 0, self.map_size - 1).astype(int)
        min_d = np.min(self.sdf_map[cy, cx])
        return bool(min_d < margin)

    def _path_danger_cost(self, A, B, margin=15.0):
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

    def _calc_soft_penalty(self, pos, min_dist_obs, obs_margin=50.0, near_crash_margin=20.0, in_signal=False):
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
        self.detected_signal_type = 0
        self._amc_cache_steps = 0
        self._amc_cache_type  = 0
        self.chosen_target = -1
        self.target_lock_steps = 0      # 목표 고정 잔여 스텝
        self.prev_dist_to_target = None # 이전 스텝의 목표까지 거리
        self.target_history = collections.deque(maxlen=2)  # 최근 선택 타겟 (핑퐁 방지)
        self._generate_world()
        return self.get_state()

    # =================================================================
    #  AMC (자동 변조 분류)
    # =================================================================
    def _load_amc(self):
        """my_amc_model.h5 와 RML2016.10a_dict.dat 로드"""
        try:
            import tensorflow as tf
            # Python 레벨 TF 로거도 ERROR만 남기도록 (환경변수와 별개로 한 번 더)
            tf.get_logger().setLevel('ERROR')
            try:
                tf.autograph.set_verbosity(0)
            except Exception:
                pass
            # absl 로깅 억제: "All log messages before absl::InitializeLog()..." 경고 제거
            try:
                from absl import logging as absl_logging
                absl_logging.set_verbosity(absl_logging.ERROR)
                absl_logging.use_absl_handler()
            except Exception:
                pass
            # compile=False: 추론만 하므로 metric 컴파일 생략 → compile_metrics 경고 제거
            self._amc_model = tf.keras.models.load_model('my_amc_model.h5', compile=False)
            print('[AMC] CNN 모델 로드 완료')
        except Exception as e:
            print(f'[AMC] 모델 로드 실패: {e}')
            self._amc_model = None

        try:
            import pickle
            with open('RML2016.10a_dict.dat', 'rb') as f:
                self._rml_raw = pickle.load(f, encoding='latin1')
            self._mods      = sorted(set(k[0] for k in self._rml_raw.keys()))
            self._snrs_list = sorted(set(k[1] for k in self._rml_raw.keys()))
            print(f'[AMC] 데이터셋 로드 완료  변조: {self._mods}')
        except Exception as e:
            print(f'[AMC] 데이터셋 로드 실패: {e}')
            self._rml_raw = None

    def _rssi_to_snr(self, rssi):
        if self._snrs_list:
            return max(self._snrs_list)  # 리스트에 있는 값 중 무조건 제일 큰 값(예: 18) 선택
        return 18

    def _classify_signal(self, mod_type, rssi):
        """
        RML2016 샘플을 AMC CNN으로 분류
        Returns: 예측 변조 방식 문자열 (모델 없으면 mod_type 그대로)
        """
        if self._amc_model is None or self._rml_raw is None:
            return mod_type   # 폴백: 완벽 분류

        target_snr = self._rssi_to_snr(rssi)
        samples = self._rml_raw.get((mod_type, target_snr))
        if samples is None:
            return mod_type

        sig = samples[np.random.randint(len(samples))]  # (2, 128)
        ai_input = sig.T[np.newaxis, ...]               # (1, 128, 2)
        # model() 직접 호출 — predict()보다 빠름
        pred = self._amc_model(ai_input, training=False).numpy()[0]
        return self._mods[int(np.argmax(pred))]

    def get_amc_rssi(self, pos):
        """
        AMC 분류 결과 기반 BPSK RSSI 반환
        - BPSK로 분류 → 실제 RSSI 반환 (드론이 찾으러 감)
        - QAM으로 분류 → 0 반환        (무시)
        Returns: (reward_bpsk_rssi, signal_type)   signal_type: 0/1/2
        """
        raw_bpsk = self.get_bpsk_rssi(pos)
        raw_qam  = max(
            (max(0.0, 1.0 - np.linalg.norm(qp - pos) / QAM_SIGNAL_RADIUS)
             for qp in self.qam_pos
             if np.linalg.norm(qp - pos) < QAM_SIGNAL_RADIUS),
            default=0.0
        )

        if raw_bpsk == 0.0 and raw_qam == 0.0:
            # 신호 범위를 벗어나도 이전 감지 상태(detected_signal_type) 유지
            return 0.0, self.detected_signal_type

        # 캐시 유효
        if self._amc_cache_steps > 0:
            self._amc_cache_steps -= 1
            self.detected_signal_type = self._amc_cache_type
            return (raw_bpsk if self._amc_cache_type == 1 else 0.0), self._amc_cache_type

        # AMC 분류 실행
        if raw_bpsk >= raw_qam:
            pred = self._classify_signal('BPSK',  raw_bpsk)
        else:
            pred = self._classify_signal('QAM16', raw_qam)

        if pred == 'BPSK':
            sig_type, reward_rssi = 1, raw_bpsk
        elif pred in ('QAM16', 'QAM64'):
            sig_type, reward_rssi = 2, 0.0
        else:
            sig_type, reward_rssi = 0, 0.0

        self._amc_cache_type  = sig_type
        self._amc_cache_steps = AMC_CLASSIFY_INTERVAL
        self.detected_signal_type = sig_type
        return reward_rssi, sig_type

    def get_bpsk_rssi(self, pos):
        dist = np.linalg.norm(self.bpsk_pos - pos)
        return max(0.0, 1.0 - dist / BPSK_SIGNAL_RADIUS) if dist < BPSK_SIGNAL_RADIUS else 0.0

    def get_state(self):
        curr_bpsk, _ = self.get_amc_rssi(self.drone_pos)
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
        
        if self._min_dist_to_obs(new_pos) < 2.0:
            return self.get_state(), -3000.0, True, "Obs Crash"

        self.drone_pos = new_pos
        self.steps    += 1
        curr_bpsk, _sig_type = self.get_amc_rssi(new_pos)
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

        if curr_bpsk == 0.0:  
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
        else: # 탐색 모드 (curr_bpsk == 0.0)
            self.search_vec = velocity.copy()

            # ── 목표 고정(lock) 로직 ──
            # chosen_target이 없거나(-1), 현재 구역과 같거나, lock이 만료됐을 때만 재계산
            # ── 타겟 재선택 조건 ──
            # 1) 타겟 없음  2) lock 만료  3) 현재 구역 == 타겟 구역
            # lock 만료 시에는 hysteresis: 현재 타겟보다 belief가 5% 이상 높아야 교체
            HYSTERESIS = 0.05
            lock_expired = self.target_lock_steps <= 0
            # [수정 B] 목표 구역에 '스치기만' 해도 lock이 풀려 재선택이 트리거되던 문제 수정.
            # 목표 구역에 도달한 뒤, 그 구역에서 최소 탐색 시간(BELIEF_FREEZE_STEPS)을
            # 채웠을 때만 도달로 인정하고 lock을 해제한다. 빙글빙글 돌며 목표 구역을
            # 잠깐 통과하는 것만으로 재평가가 일어나는 출렁임을 줄인다.
            entered_target = (self.chosen_target == quad_idx) and (steps_here > BELIEF_FREEZE_STEPS)
            if entered_target:
                self.target_lock_steps = 0

            if lock_expired and self.chosen_target != -1 and not entered_target:
                curr_target_belief = self.belief[self.chosen_target]
                best_other = max(
                    (self.belief[i] for i in range(self.macro_count)
                     if i != quad_idx and i != self.chosen_target),
                    default=0.0
                )
                should_switch = best_other > curr_target_belief + HYSTERESIS
            else:
                should_switch = False

            need_retarget = (
                self.chosen_target == -1
                or entered_target
                or should_switch
            )

            if need_retarget:
                temp_belief = self.belief.copy()
                temp_belief[quad_idx] = -1.0  # 현재 구역 제외
                # 최근 방문한 타겟도 제외 (핑퐁 방지)
                for prev_t in self.target_history:
                    if prev_t != quad_idx:
                        temp_belief[prev_t] = max(temp_belief[prev_t] - 0.15, -1.0)

                max_p = float(np.max(temp_belief))
                candidates = np.where(np.isclose(temp_belief, max_p, atol=1e-5))[0]

                if len(candidates) > 1:
                    target_centers = self.macro_centers[candidates]
                    # [수정 A] 경로 위험도 평가의 출발점을 드론의 실시간 위치가 아니라
                    # '현재 구역의 중심'으로 고정한다. 드론이 구역 안에서 빙글빙글 돌며
                    # 위치가 미세하게 바뀌어도 동점 평가 결과가 흔들리지 않게 하여
                    # 목표 구역이 계속 바뀌는(출렁임) 현상을 막는다.
                    eval_origin = self.macro_centers[quad_idx]
                    danger_costs = np.array([self._path_danger_cost(eval_origin, tc) for tc in target_centers])
                    dest_danger = np.array([max(0.0, 80.0 - self._min_dist_to_obs(tc)) for tc in target_centers])
                    total_costs  = danger_costs * 6.0 + dest_danger * 15.0  # 장애물 기피 완화
                    best_target  = int(candidates[np.argmin(total_costs)])
                else:
                    best_target = int(candidates[0])

                if self.chosen_target != -1:
                    self.target_history.append(self.chosen_target)
                self.chosen_target = best_target
                self.target_lock_steps = 20  # 최소 20스텝 고정
                self.prev_dist_to_target = None  # 목표 바뀌면 거리 기준 초기화
            else:
                best_target = self.chosen_target
                self.target_lock_steps -= 1  # 잔여 고정 스텝 차감

            # 목표 방향으로 드론을 유도하는 로직
            target_pos    = self.macro_centers[best_target]
            diff_vec      = target_pos - self.drone_pos
            dist_to_goal  = np.linalg.norm(diff_vec)
            dir_to_target = diff_vec / (dist_to_goal + 1e-8)

            is_in_zero_zone = self.belief[quad_idx] < 0.01
            escape_weight = 2.5 if is_in_zero_zone else 1.0

            belief_diff  = float(np.clip(self.belief[best_target] - self.belief[quad_idx], 0.0, 0.5))
            dot_val = float(np.dot(velocity, dir_to_target))

            multiplier = (150.0 + belief_diff * 600.0) * escape_weight
            drive_reward = dot_val * multiplier

            # ── 거리 감소 보상 ──
            # 가까워진 만큼 보상, 멀어지면 패널티
            # dist_scale: 멀리 있을수록 보상을 키워 강하게 유도
            if self.prev_dist_to_target is not None:
                dist_delta = self.prev_dist_to_target - dist_to_goal  # 양수 = 가까워짐
                dist_scale = float(np.clip(dist_to_goal / (self.map_size * 0.5), 0.3, 1.5))
                approach_reward = dist_delta * 40.0 * dist_scale
            else:
                approach_reward = 0.0
            self.prev_dist_to_target = dist_to_goal

            reward = step_penalty + drive_reward + approach_reward + soft_penalty + macro_crossing_bonus

        self.prev_bpsk_rssi = curr_bpsk

        # ── 타임아웃 처리 ──
        # MAX_STEPS를 다 쓰고도 목표(Target Found)에 도달하지 못한 채 끝나면
        # "그냥 안전하게 떠다니며 시간 버티기"가 이득이 되지 않도록 패널티를 준다.
        # (목표 도달은 위에서 +5000으로 이미 종료 처리되므로 여기 도달하면 미도달 = 실패)
        if self.steps >= MAX_STEPS:
            reward += -1500.0
            return self.get_state(), reward, True, "Timeout"

        return self.get_state(), reward, False, "Normal"