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

AMC_CLASSIFY_INTERVAL = 5    # 한 번 감지하면 스텝 동안 결과 고정!

BELIEF_DECAY        = 0.95
BELIEF_FREEZE_STEPS = int(MAP_SIZE * 0.01)   

_QAM_POS = np.array([[400.0, 200.0], [200.0, 400.0]])

# ── 라이다(LiDAR) 설정 ──
NUM_LIDAR_RAYS  = 16      
LIDAR_MAX_RANGE = 100.0   

# ── 커버리지 맵 설정 ──
# 드론이 탐색한 영역을 색칠로 기록하는 격자(70x70, 한 칸 = 10x10px).
# 값: 0=미탐색, 1=탐색됨(드론이 지나가며 본 영역), 2=장애물 감지 지점
COVERAGE_N = 70
COVERAGE_REWARD_SCALE = 5   # 색칠 보상 스케일(새 칸 × belief지수가중 × 이 값)
# 색칠 반경: 드론이 한 스텝에 칠하는 범위. LiDAR(100)보다 살짝 크게.
# (신호 감지 범위 250과는 별개 — 색칠은 '드론이 둘러본 영역', 신호는 '목표 감지')
COVERAGE_PAINT_RADIUS = 120.0


class DroneEnv:
    def __init__(self):
        self.map_size    = MAP_SIZE
        self.drone_start = START_POS.copy()
        
        self.MACRO_N     = 3
        self.macro_count = self.MACRO_N * self.MACRO_N
        self.state_dim   = 14 + 3 * self.macro_count + NUM_LIDAR_RAYS
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
        # 캐시/버전 기본값(reset 전 안전장치)
        self._belief_version = 0
        self._lidar_cache = None
        self._adjbelief_cache = None
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
        # [최적화] 같은 위치에 대한 LiDAR 결과를 캐시한다.
        # 한 스텝 안에서 get_state()/_adjusted_belief() 등이 동일한 drone_pos로
        # 여러 번 호출돼도 ray casting(30회 루프)을 한 번만 수행하게 한다.
        key = (float(pos[0]), float(pos[1]))
        cached = getattr(self, "_lidar_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        t = np.zeros(NUM_LIDAR_RAYS, dtype=np.float32)
        for _ in range(30):
            curr_pos = pos + t[:, None] * self._lidar_dirs
            cx = np.clip(curr_pos[:, 0], 0, self.map_size - 1).astype(int)
            cy = np.clip(curr_pos[:, 1], 0, self.map_size - 1).astype(int)
            dists = self.sdf_map[cy, cx]
            t += dists
            if np.all((dists < 1.0) | (t >= LIDAR_MAX_RANGE)):
                break
        result = np.clip(t / LIDAR_MAX_RANGE, 0.0, 1.0)
        self._lidar_cache = (key, result)
        return result

    def _lidar_openness_toward(self, lidar, direction):
        """주어진 방향(단위벡터)을 향하는 LiDAR 개방도(0~1)를 반환.
        드론이 '지금 센서로 보이는' 정보만 사용한다는 가정에 부합하도록,
        SDF로 경로 전체를 스캔하지 않고 LiDAR 광선 측정값만 이용한다.

        - 방향과 각 LiDAR 광선의 정렬도(내적)를 구해 가장 정렬된 광선을 찾고,
        - 16방향(22.5° 간격)의 각도 오차를 보완하기 위해 정렬 상위 인접 3개 광선의
          최솟값(가장 막힌 값)을 보수적으로 취한다.
        반환값이 클수록 그 방향이 뚫려 있음(장애물 멀거나 없음)을 의미한다.
        lidar: _cast_lidar_rays() 결과(정규화 거리 0~1, 길이 NUM_LIDAR_RAYS)."""
        dot = self._lidar_dirs @ direction          # 각 광선과의 정렬도
        top = np.argsort(dot)[-3:]                   # 가장 정렬된 상위 3개 광선
        return float(np.min(lidar[top]))             # 보수적으로 최솟값(가장 막힌 값)

    def _paint_coverage(self):
        """드론의 현재 위치에서 센서 관측을 커버리지 맵(70x70)에 색칠한다.
        - 색칠 반경(COVERAGE_PAINT_RADIUS) 안의 칸 → 1(탐색됨)로 칠함
        - LiDAR가 장애물에 닿은 지점의 칸 → 2(장애물)로 칠함
        값이 누적(0→1→2)되므로 같은 곳을 다시 지나도 추가 변화가 없어
        '중복 감소'가 원천적으로 생기지 않는다.

        반환: 이번 스텝에 '새로 칠해진 칸 수'를 3x3 구역별로 집계한 배열(9,).
              색칠 보상 계산에 쓰인다(이미 칠한 칸은 0이므로 중복 보상 없음).
        """
        cell_px = self.map_size / COVERAGE_N   # 한 칸의 픽셀 크기(=10px)
        cells_per_macro = COVERAGE_N // self.MACRO_N

        # ── (1) 색칠 반경 안을 1로 색칠 ──
        R = COVERAGE_PAINT_RADIUS
        idx = (np.arange(COVERAGE_N) + 0.5) * cell_px
        gx, gy = np.meshgrid(idx, idx)
        dist2 = (gx - self.drone_pos[0])**2 + (gy - self.drone_pos[1])**2
        within = dist2 <= (R * R)
        # 아직 0(미탐색)인 칸만 새로 칠함 → 이게 '새 칸'
        paint_mask = within & (self.coverage == 0)
        self.coverage[paint_mask] = 1

        # 새로 칠해진 칸을 3x3 구역별로 집계
        new_cells = np.zeros(self.macro_count, dtype=np.float32)
        if np.any(paint_mask):
            ys, xs = np.where(paint_mask)
            macro_y = np.clip(ys // cells_per_macro, 0, self.MACRO_N - 1)
            macro_x = np.clip(xs // cells_per_macro, 0, self.MACRO_N - 1)
            macro_idx = macro_y * self.MACRO_N + macro_x
            for mi in macro_idx:
                new_cells[mi] += 1.0

        # ── (2) LiDAR가 장애물에 닿은 지점을 2로 색칠 ──
        lidar = self._cast_lidar_rays(self.drone_pos)
        for r in range(NUM_LIDAR_RAYS):
            d_px = lidar[r] * LIDAR_MAX_RANGE
            if d_px < LIDAR_MAX_RANGE - 1.0:
                hit = self.drone_pos + self._lidar_dirs[r] * d_px
                cx = int(np.clip(hit[0] / cell_px, 0, COVERAGE_N - 1))
                cy = int(np.clip(hit[1] / cell_px, 0, COVERAGE_N - 1))
                self.coverage[cy, cx] = 2

        return new_cells

    def _update_belief_from_coverage(self):
        """커버리지 맵을 바탕으로 각 3x3 구역의 belief를 '직접 설정'한다.
        매 스텝 곱셈 누적이 아니라, 현재 커버리지 상태로부터 belief를 재계산하므로
        같은 곳을 맴돌아도 belief가 무한히 깎이지 않는다(중복 감소 없음).

        구역별 신호 점수:
          score = (1 - explored_frac)          # 탐색된 비율만큼 목표 확률 감소
          if 장애물 칸 있으면 score *= 0.9      # 장애물 구역은 추가로 10% 더 감소
        이 점수를 정규화해 belief로 삼는다(탐색 안 된 구역일수록 높음).
        """
        cells_per_macro = COVERAGE_N // self.MACRO_N   # 70//3 = 23칸
        score = np.zeros(self.macro_count, dtype=np.float32)

        for gy in range(self.MACRO_N):
            for gx in range(self.MACRO_N):
                gi = gy * self.MACRO_N + gx
                # 이 구역에 해당하는 커버리지 칸 블록
                y0, y1 = gy * cells_per_macro, (gy + 1) * cells_per_macro
                x0, x1 = gx * cells_per_macro, (gx + 1) * cells_per_macro
                block = self.coverage[y0:y1, x0:x1]
                total = block.size

                explored = np.count_nonzero(block >= 1)   # 1 또는 2 = 탐색됨
                has_obstacle = np.any(block == 2)

                explored_frac = explored / total          # 0~1
                s = 1.0 - explored_frac                    # 탐색될수록 목표확률 감소
                if has_obstacle:
                    s *= 0.9                               # 장애물 구역 추가 10% 감소
                score[gi] = s

        # 점수를 belief 확률 분포로 정규화
        score = np.clip(score, 1e-6, None)
        self.belief = (score / np.sum(score)).astype(np.float32)
        self._belief_version += 1

    def _adjusted_belief(self):
        """정책망 입력(state) 및 목표 선택에 쓰이는 belief를 반환한다.
        belief 본체(self.belief)가 이미 _bayes_belief_update()로 신호·장애물 관측을
        누적 반영하므로, 별도의 위치 종속 보정 레이어를 더하지 않는다.
        (예전에는 매 스텝 거리·개방도 보정을 얹었으나, 드론 위치에 따라 belief가
         출렁이는 문제가 있어 제거했다. 이제 belief는 관측 누적값이라 안정적이다.)"""
        return self.belief

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
    
    def _calc_soft_penalty(self, pos, min_dist_obs, obs_margin=30.0, near_crash_margin=10.0, in_signal=False): 
        scale = 2.0 if in_signal else 1.0
        soft = 0.0
        if min_dist_obs < obs_margin:
            soft -= ((obs_margin - min_dist_obs) / obs_margin) ** 2 * 150.0 * scale
        if min_dist_obs < near_crash_margin:
            soft -= 200.0 * scale
        return soft

    def reset(self):
        self.drone_pos   = self.drone_start.copy()
        self.prev_action = np.array([1.0, 1.0]) / np.sqrt(2)
        self.search_vec  = self.prev_action.copy()
        self.steps       = 0
        self.prev_bpsk_rssi = 0.0
        self.belief = np.full(self.macro_count, 1.0 / self.macro_count, dtype=np.float32)
        self._belief_version = 0          # belief 본체가 바뀔 때마다 증가(캐시 무효화용)
        self._lidar_cache = None
        self._adjbelief_cache = None
        # 커버리지 맵: 0=미탐색, 1=탐색됨, 2=장애물 감지. 매 에피소드 초기화.
        self.coverage = np.zeros((COVERAGE_N, COVERAGE_N), dtype=np.int8)
        self.area_visit_steps = np.zeros(self.macro_count, dtype=int)
        self.current_macro = -1
        self.macro_entry_freeze = 0
        self.visited_quad_history = collections.deque(maxlen=self.macro_count)
        self.detected_signal_type = 0
        self._amc_cache_steps = 0
        self._amc_cache_type  = 0
        self.chosen_target = -1         # (시각화/호환용; 보상 로직에선 미사용)
        self.prev_dist_to_macros = None # 이전 스텝의 각 구역 중심까지 거리(9,) — belief 가중 전진용
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
        except Exception as e:
            print(f'[AMC] 모델 로드 실패: {e}')
            self._amc_model = None

        try:
            import pickle
            with open('RML2016.10a_dict.dat', 'rb') as f:
                self._rml_raw = pickle.load(f, encoding='latin1')
            self._mods      = sorted(set(k[0] for k in self._rml_raw.keys()))
            self._snrs_list = sorted(set(k[1] for k in self._rml_raw.keys()))
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
        
        # belief를 3x9로: 각 구역의 (드론 기준 상대방향 x, y, 확률).
        # 정책망이 '확률 높은 구역이 내 기준 어느 방향인지'를 직접 보고 판단한다.
        # 상대방향은 드론 위치 기준이라 좌표 변환 없이 바로 쓸 수 있다.
        belief = self.belief
        vecs = self.macro_centers - self.drone_pos[None, :]      # (9,2)
        dists = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        dirs = vecs / dists                                       # 단위 방향벡터 (9,2)
        for gi in range(self.macro_count):
            state.append(float(dirs[gi, 0]))   # 상대방향 x
            state.append(float(dirs[gi, 1]))   # 상대방향 y
            state.append(float(belief[gi]))    # 그 구역 확률
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
            return self.get_state(), -1500.0, True, "Wall Crash"
        
        if self._min_dist_to_obs(new_pos) < 2.0:
            return self.get_state(), -2000.0, True, "Obs Crash"

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
                macro_crossing_bonus = 250.0 # 새 구역 처음 진입 보너스
            self.current_macro = quad_idx
            self.chosen_target = -1 
            self.visited_quad_history.append(quad_idx)

        self.area_visit_steps[quad_idx] += 1
        steps_here = self.area_visit_steps[quad_idx]

        if steps_here <= BELIEF_FREEZE_STEPS:
            self.macro_entry_freeze = 1
        else:
            self.macro_entry_freeze = 0

        # 색칠 보상은 '칠하기 직전' belief로 계산해야 한다.
        # (색칠 후 belief를 재계산하면 그 구역 belief가 떨어지므로, 칠한 보람이 사라짐)
        belief_before = self.belief.copy()

        # 커버리지 색칠은 신호 유무와 무관하게 매 스텝 수행. 새로 칠한 칸을 구역별로 받음.
        new_cells = self._paint_coverage()

        if curr_bpsk == 0.0:
            # 신호 없음 → 커버리지 기반으로 belief 재계산(무한감소 없음).
            self._update_belief_from_coverage()

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

            # ── 색칠 보상 (belief 지수 가중) ───────────────────────────
            # belief 높은 구역의 '새 칸'을 칠할수록 큰 보상.
            # 방향/거리를 쓰지 않고 '결과(색칠)'에 보상하므로 목표 출렁임이 없다.
            # 동점이어도 둘 중 아무 구역이나 칠하면 보상 → 가까운 곳부터 칠하러 감.
            #
            # belief 지수 가중: w = exp(10*(belief - 0.13))  
            #   기준점 0.13(거의 균등=1/9≈0.11)이라, belief가 균등한 초·중반에도
            #   보상이 미지근하지 않게 한다(과감성 확보). 높은 구역은 여전히 급증.
            #   - belief 11% → w = 0.8 , 13% → 1.0(기준), 20% → 2, 30% → 5.5, 40% → 14.9
            weight = np.exp(10.0 * (belief_before - 0.13))   
            paint_reward = float(np.sum(new_cells * weight)) * COVERAGE_REWARD_SCALE 

            reward = step_penalty + paint_reward + soft_penalty + macro_crossing_bonus

        self.prev_bpsk_rssi = curr_bpsk

        # ── 타임아웃 처리 ──
        # MAX_STEPS를 다 쓰고도 목표(Target Found)에 도달하지 못한 채 끝나면
        # "그냥 안전하게 떠다니며 시간 버티기"가 이득이 되지 않도록 패널티를 준다.
        # (목표 도달은 위에서 +5000으로 이미 종료 처리되므로 여기 도달하면 미도달 = 실패)
        if self.steps >= MAX_STEPS:
            reward += -2000.0
            return self.get_state(), reward, True, "Timeout"

        return self.get_state(), reward, False, "Normal"