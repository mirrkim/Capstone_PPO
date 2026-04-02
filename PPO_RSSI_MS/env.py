import numpy as np

# =============================================
# --- 환경 설정 상수 ---
# =============================================
MAP_SIZE         = 700
START_POS        = np.array([30.0, 30.0])

OBS_COUNT        = 5
OBS_RADIUS       = 40

BPSK_SIGNAL_RADIUS  = 250
QAM_SIGNAL_RADIUS   = 150

TARGET_RSSI_THRESHOLD = 0.85
MAX_STEPS = 500

# =============================================
# 🎯 신호원 생성 모드 설정 패널
# =============================================
TARGET_MODE = 'random'

MANUAL_BPSK_POS  = np.array([500.0, 550.0])
MANUAL_QAM0_POS  = np.array([200.0, 500.0])
MANUAL_QAM1_POS  = np.array([550.0, 200.0])
# =============================================

OBS_SOFT_MARGIN = 100
OBS_HARD_MARGIN = 55


class DroneEnv:
    """
    700×700 2D 드론 환경

    ── 설계 원칙 (물리적 제약 엄수) ────────────────────────────────
    [탐색 모드 - BPSK 신호 반경 밖]
      드론은 BPSK 신호를 전혀 수신하지 못한다.
      따라서 드론이 알 수 있는 정보는:
        - 자신의 위치 (GPS)
        - 이전 스텝의 RSSI = 0 (변화량도 0)
        - 이전 행동 벡터 (관성)
        - search_vec (내부 탐색 방향, BPSK 좌표와 무관)
        - 장애물 상대 위치 (센서)
      절대 사용 불가: BPSK 좌표, 실제 거리, 실제 방향

    [추적 모드 - BPSK 신호 반경 안]
      드론은 BPSK 신호를 수신한다.
        - RSSI 값 (신호 세기)
        - RSSI 변화량 (이동 전후 차이)
        - AOA (Angle of Arrival): 신호 도래각 → 신호원 방향 벡터
      AOA로 신호원 방향은 알 수 있으나 거리는 모른다.
      dist_diff(실제 거리 변화)는 사용 불가.

    [장애물 회피]
      - APF 척력: velocity를 실시간 보정 (action과 독립적)
      - 소프트 패널티: 근접 시 reward 감점

    ── state 구성 (총 29차원) ──────────────────────────────────────
     0- 1  : 드론 위치 정규화 (x/MAP, y/MAP)        ← GPS
     2- 3  : 맵 끝까지 남은 거리 정규화              ← GPS
     4     : BPSK RSSI (반경 밖=0)                  ← 수신기
     5     : BPSK RSSI 변화량 ×100                  ← 수신기
     6- 7  : 이전 행동 벡터 (관성)                   ← 내부 상태
     8- 9  : sensor_dir
              반경 안 → AOA 방향벡터                 ← AOA 센서
              반경 밖 → search_vec (내부 탐색벡터)   ← 내부 상태
    10-11  : 64QAM #0 RSSI, 변화량 ×100             ← 수신기
    12-13  : 64QAM #1 RSSI, 변화량 ×100             ← 수신기
    14-28  : 장애물 5개 × (rx, ry, dist) = 15차원   ← 거리 센서
    ────────────────────────────────────────────────────────────────
    """

    def __init__(self):
        self.map_size    = MAP_SIZE
        self.drone_start = START_POS.copy()
        self.state_dim   = 29
        self.action_dim  = 2

        self.obstacles = [
            np.array([200.0, 200.0]),
            np.array([450.0, 150.0]),
            np.array([150.0, 480.0]),
            np.array([500.0, 450.0]),
            np.array([330.0, 350.0]),
        ]

    # ──────────────────────────────────────────────────────────────
    def reset(self):
        self.drone_pos   = self.drone_start.copy()
        self.prev_action = np.array([1.0, 1.0]) / np.sqrt(2)
        self.search_vec  = np.array([1.0, 1.0]) / np.sqrt(2)
        self.steps       = 0

        self.prev_bpsk_rssi = 0.0
        self.prev_qam0_rssi = 0.0
        self.prev_qam1_rssi = 0.0

        # 방문 격자 (14×14, 셀 크기 50px) - 탐색 효율용
        self.GRID_N      = 14
        self.visit_count = np.zeros((self.GRID_N, self.GRID_N), dtype=np.int32)

        if TARGET_MODE == 'manual':
            self.bpsk_pos = MANUAL_BPSK_POS.copy()
            self.qam_pos  = [MANUAL_QAM0_POS.copy(), MANUAL_QAM1_POS.copy()]
        else:
            self.bpsk_pos = self._spawn_signal(
                min_gap_obs=OBS_RADIUS + 30, min_gap_start=220, others=[])
            qam0 = self._spawn_signal(
                min_gap_obs=OBS_RADIUS + 20, min_gap_start=150,
                others=[self.bpsk_pos], min_gap_others=QAM_SIGNAL_RADIUS + 50)
            qam1 = self._spawn_signal(
                min_gap_obs=OBS_RADIUS + 20, min_gap_start=150,
                others=[self.bpsk_pos, qam0], min_gap_others=QAM_SIGNAL_RADIUS + 50)
            self.qam_pos = [qam0, qam1]

        # ※ prev_dist 제거: 신호 반경 밖에서 실제 거리를 드론이 알 수 없음
        #   추적 모드(신호 안)에서는 dist_diff 대신 RSSI 변화량을 사용
        return self.get_state()

    def _spawn_signal(self, min_gap_obs, min_gap_start, others, min_gap_others=0):
        while True:
            pos = np.random.uniform(50, self.map_size - 50, size=2)
            if pos[0] <= min_gap_start and pos[1] <= min_gap_start:
                continue
            if any(np.linalg.norm(pos - obs) < min_gap_obs for obs in self.obstacles):
                continue
            if others and min_gap_others > 0:
                if any(np.linalg.norm(pos - o) < min_gap_others for o in others):
                    continue
            return pos

    # ──────────────────────────────────────────────────────────────
    def get_bpsk_rssi(self, pos):
        """BPSK RSSI: 반경 안에서만 0 초과, 반경 밖은 정확히 0.0"""
        dist = np.linalg.norm(self.bpsk_pos - pos)
        if dist < BPSK_SIGNAL_RADIUS:
            return 1.0 - dist / BPSK_SIGNAL_RADIUS
        return 0.0

    def get_qam_rssi(self, pos, idx):
        dist = np.linalg.norm(self.qam_pos[idx] - pos)
        if dist < QAM_SIGNAL_RADIUS:
            return 1.0 - dist / QAM_SIGNAL_RADIUS
        return 0.0

    # ──────────────────────────────────────────────────────────────
    def get_state(self):
        curr_bpsk = self.get_bpsk_rssi(self.drone_pos)
        curr_qam0 = self.get_qam_rssi(self.drone_pos, 0)
        curr_qam1 = self.get_qam_rssi(self.drone_pos, 1)

        bpsk_diff = curr_bpsk - self.prev_bpsk_rssi
        qam0_diff = curr_qam0 - self.prev_qam0_rssi
        qam1_diff = curr_qam1 - self.prev_qam1_rssi

        if curr_bpsk > 0.0:
            # ── 추적 모드: AOA로 신호원 방향 측정 ─────────────────
            # 실제 시스템에서 AOA 센서가 제공하는 값을 시뮬레이션
            # (신호 수신 중에만 유효)
            vec = self.bpsk_pos - self.drone_pos
            n   = np.linalg.norm(vec)
            sensor_dir = vec / n if n > 0.001 else np.zeros(2)
        else:
            # ── 탐색 모드: search_vec (내부 상태, BPSK 좌표 무관) ──
            sensor_dir = self.search_vec

        state = [
            self.drone_pos[0] / self.map_size,          # GPS
            self.drone_pos[1] / self.map_size,
            (self.map_size - self.drone_pos[0]) / self.map_size,
            (self.map_size - self.drone_pos[1]) / self.map_size,
            curr_bpsk,                                   # RSSI (밖=0)
            bpsk_diff * 100.0,                           # RSSI 변화량
            self.prev_action[0],                         # 관성
            self.prev_action[1],
            sensor_dir[0],                               # AOA or search_vec
            sensor_dir[1],
            curr_qam0,
            qam0_diff * 100.0,
            curr_qam1,
            qam1_diff * 100.0,
        ]

        for obs in self.obstacles:
            rx   = (obs[0] - self.drone_pos[0]) / self.map_size
            ry   = (obs[1] - self.drone_pos[1]) / self.map_size
            dist = np.linalg.norm([rx, ry])
            state.extend([rx, ry, dist])

        return np.array(state, dtype=np.float32)  # 29차원

    # ──────────────────────────────────────────────────────────────
    def _apf_repulsion(self, pos, velocity):
        """
        APF 척력 - 접선 편향 방식
        장애물이 드론을 밀어내되, 정면 충돌 시 접선 방향으로 편향시켜
        우회 경로를 유도한다.
        """
        APF_INFLUENCE = 100.0
        APF_K_REP     = 90000.0
        APF_MAX       = 0.75

        correction = np.zeros(2)

        for obs in self.obstacles:
            d = np.linalg.norm(pos - obs)
            if d >= APF_INFLUENCE or d < 1.0:
                continue

            rep_dir   = (pos - obs) / d
            magnitude = APF_K_REP * (1.0/d - 1.0/APF_INFLUENCE) / (d * d)

            normal_proj = np.dot(velocity, rep_dir)
            normal_vec  = rep_dir * normal_proj
            tangent_vec = velocity - normal_vec

            if normal_proj < 0:
                # 장애물 방향 이동: 접선 편향 강화
                tangent_n = np.linalg.norm(tangent_vec)
                if tangent_n > 0.001:
                    tangent_boost = tangent_vec / tangent_n * magnitude * 2.0
                else:
                    perp = np.array([-rep_dir[1], rep_dir[0]])
                    tangent_boost = perp * magnitude * 2.0
                correction += rep_dir * magnitude + tangent_boost
            else:
                correction += rep_dir * magnitude * 0.5

        # ── 벽 척력 (장애물과 동일 원리) ────────────────────────
        WALL_INFLUENCE = 80.0    # 벽 척력 영향 거리
        WALL_K         = 60000.0 # 벽 척력 계수
        wall_reps = [
            (pos[0],                  np.array([ 1.0, 0.0])),  # 왼쪽 벽
            (self.map_size - pos[0],  np.array([-1.0, 0.0])),  # 오른쪽 벽
            (pos[1],                  np.array([ 0.0, 1.0])),  # 아래쪽 벽
            (self.map_size - pos[1],  np.array([ 0.0,-1.0])),  # 위쪽 벽
        ]
        for d_wall, rep_dir in wall_reps:
            if d_wall >= WALL_INFLUENCE or d_wall < 1.0:
                continue
            mag = WALL_K * (1.0/d_wall - 1.0/WALL_INFLUENCE) / (d_wall * d_wall)
            normal_proj = np.dot(velocity, rep_dir)
            normal_vec  = rep_dir * normal_proj
            tangent_vec = velocity - normal_vec
            if normal_proj < 0:
                t_n = np.linalg.norm(tangent_vec)
                if t_n > 0.001:
                    correction += rep_dir * mag + (tangent_vec/t_n) * mag * 1.5
                else:
                    correction += rep_dir * mag * 2.0
            else:
                correction += rep_dir * mag * 0.3

        c_norm = np.linalg.norm(correction)
        if c_norm > APF_MAX:
            correction = correction / c_norm * APF_MAX

        return correction

    # ──────────────────────────────────────────────────────────────
    def _obs_penalty(self, pos):
        """2단계 장애물 소프트 패널티"""
        penalty = 0.0
        for obs in self.obstacles:
            d = np.linalg.norm(pos - obs)
            if OBS_HARD_MARGIN < d < OBS_SOFT_MARGIN:
                t = (OBS_SOFT_MARGIN - d) / (OBS_SOFT_MARGIN - OBS_HARD_MARGIN)
                penalty -= t * 200.0
            elif OBS_RADIUS < d <= OBS_HARD_MARGIN:
                t = (OBS_HARD_MARGIN - d) / (OBS_HARD_MARGIN - OBS_RADIUS)
                penalty -= t * t * 800.0
        return penalty

    # ──────────────────────────────────────────────────────────────
    def step(self, action):
        move_step = 10.0

        action   = np.clip(action, -1.0, 1.0)
        a_norm   = np.linalg.norm(action)
        action   = action / a_norm if a_norm > 0.001 else self.prev_action

        inertia  = 0.5
        velocity = action * (1.0 - inertia) + self.prev_action * inertia
        v_norm   = np.linalg.norm(velocity)
        velocity = velocity / v_norm if v_norm > 0.001 else self.prev_action

        # ── APF 척력 보정 (물리적 회피, 보상과 독립) ──────────────
        apf_rep = self._apf_repulsion(self.drone_pos, velocity)
        if np.linalg.norm(apf_rep) > 0.001:
            velocity = velocity + apf_rep
            v_norm2  = np.linalg.norm(velocity)
            if v_norm2 > 0.001:
                velocity = velocity / v_norm2

        self.prev_action = velocity
        new_pos = self.drone_pos + velocity * move_step

        # ── search_vec 벽 반사 (순수 당구공, BPSK 정보 없음) ──────
        # margin을 70px로 확대: 벽에 충분히 가까워지기 전에 방향 전환
        margin = 70.0
        if new_pos[0] < margin and self.search_vec[0] < 0:
            self.search_vec[0] *= -1
        elif new_pos[0] > self.map_size - margin and self.search_vec[0] > 0:
            self.search_vec[0] *= -1
        if new_pos[1] < margin and self.search_vec[1] < 0:
            self.search_vec[1] *= -1
        elif new_pos[1] > self.map_size - margin and self.search_vec[1] > 0:
            self.search_vec[1] *= -1
        self.search_vec /= np.linalg.norm(self.search_vec)

        # ── RSSI 계산 (수신기 측정값) ─────────────────────────────
        curr_bpsk = self.get_bpsk_rssi(new_pos)
        bpsk_diff = curr_bpsk - self.prev_bpsk_rssi  # 드론이 실제로 측정 가능

        reward = 0.0
        done   = False
        info   = ""

        # ── 벽 충돌 ────────────────────────────────────────────────
        if (new_pos[0] < 0 or new_pos[0] > self.map_size or
                new_pos[1] < 0 or new_pos[1] > self.map_size):
            reward = -2000.0
            done   = True
            info   = "Wall Crash"
            self._update_prev(curr_bpsk)
            return self.get_state(), reward, done, info

        # ── 장애물 충돌 (즉시 종료) ────────────────────────────────
        for obs in self.obstacles:
            if np.linalg.norm(new_pos - obs) <= OBS_RADIUS:
                reward = -2000.0
                done   = True
                info   = "Obstacle Crash"
                self._update_prev(curr_bpsk)
                return self.get_state(), reward, done, info

        obs_penalty  = self._obs_penalty(new_pos)

        # ── 벽 근접 패널티 ─────────────────────────────────────
        # w_hard: 이 거리 안에서는 강한 이차 패널티 (APF 보완)
        # w_soft: 이 거리 안에서는 선형 패널티 시작
        wall_penalty = 0.0
        w_soft, w_hard = 60.0, 25.0
        for coord, size in [(new_pos[0], self.map_size), (new_pos[1], self.map_size)]:
            # 가까운 벽까지의 거리
            d_near = min(coord, size - coord)
            if d_near < w_hard:
                t = (w_hard - d_near) / w_hard
                wall_penalty -= t * t * 300.0   # 최대 -300 (이차)
            elif d_near < w_soft:
                t = (w_soft - d_near) / (w_soft - w_hard)
                wall_penalty -= t * 60.0         # 최대 -60 (선형)

        self.drone_pos = new_pos
        self.steps    += 1

        # ── 목표 도달 ──────────────────────────────────────────────
        if curr_bpsk >= TARGET_RSSI_THRESHOLD:
            reward = 3000.0
            done   = True
            info   = "Goal Reached"

        else:
            if curr_bpsk > 0.0:
                # ════════════════════════════════════════════════════
                # 추적 모드 (BPSK 신호 수신 중)
                # 드론이 아는 정보: RSSI, RSSI 변화량, AOA 방향
                # ════════════════════════════════════════════════════
                step_penalty = -1.0

                # AOA로 신호원 방향 파악 → 그 방향으로 이동 시 보상
                vec  = self.bpsk_pos - self.drone_pos
                n    = np.linalg.norm(vec)
                aoa_dir   = vec / n if n > 0.001 else np.zeros(2)
                alignment = np.dot(velocity, aoa_dir)

                # RSSI 변화량 보상: 신호가 강해지는 방향으로 이동 장려
                # (드론이 실제 측정 가능한 값)
                rssi_reward = bpsk_diff * 300.0

                # 장애물 회피 방향 보상
                obs_avoid_bonus = 0.0
                for obs in self.obstacles:
                    d = np.linalg.norm(self.drone_pos - obs)
                    if d < OBS_SOFT_MARGIN:
                        away_dir = self.drone_pos - obs
                        away_n   = np.linalg.norm(away_dir)
                        if away_n > 0.001:
                            away_dir /= away_n
                        avoid_align = np.dot(velocity, away_dir)
                        proximity   = (OBS_SOFT_MARGIN - d) / OBS_SOFT_MARGIN
                        obs_avoid_bonus += avoid_align * proximity * 30.0

                reward = (step_penalty
                          + alignment       *  8.0  # AOA 방향 정렬
                          + rssi_reward             # RSSI 증가 보상
                          + obs_avoid_bonus         # 장애물 회피 방향
                          + obs_penalty
                          + wall_penalty)

            else:
                # ════════════════════════════════════════════════════
                # 탐색 모드 (신호 미수신)
                # 드론이 아는 정보: 위치(GPS), 이전행동, search_vec만
                # BPSK 좌표/거리/방향 절대 사용 불가
                # ════════════════════════════════════════════════════
                step_penalty = -0.5

                # search_vec 방향 정렬 보상 (순수 당구공 탐색)
                alignment = np.dot(velocity, self.search_vec)

                # 재방문 패널티 (루프 억제, GPS 기반 → 합법)
                gx = int(np.clip(new_pos[0] / self.map_size * self.GRID_N,
                                 0, self.GRID_N - 1))
                gy = int(np.clip(new_pos[1] / self.map_size * self.GRID_N,
                                 0, self.GRID_N - 1))
                visit_penalty = -self.visit_count[gx, gy] * 1.5
                self.visit_count[gx, gy] += 1

                reward = (step_penalty
                          + alignment   * 10.0  # search_vec 정렬
                          + visit_penalty        # 재방문 억제
                          + obs_penalty
                          + wall_penalty)

        if self.steps >= MAX_STEPS:
            done = True
            info = "Timeout"

        self._update_prev(curr_bpsk)
        return self.get_state(), reward, done, info

    # ──────────────────────────────────────────────────────────────
    def _update_prev(self, curr_bpsk):
        self.prev_bpsk_rssi = curr_bpsk
        self.prev_qam0_rssi = self.get_qam_rssi(self.drone_pos, 0)
        self.prev_qam1_rssi = self.get_qam_rssi(self.drone_pos, 1)