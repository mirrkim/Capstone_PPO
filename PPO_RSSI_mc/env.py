import numpy as np

# =============================================
# --- 환경 설정 상수 ---
# =============================================
MAP_SIZE         = 700
START_POS        = np.array([30.0, 30.0])

MAX_OBS_COUNT    = 5
OBS_RADIUS_MIN   = 50.0   # [수정 가능] 랜덤 장애물 최소 반지름
OBS_RADIUS_MAX   = 100.0  # [수정 가능] 랜덤 장애물 최대 반지름
OBS_RADIUS       = 50     # test.py 호환용

BPSK_SIGNAL_RADIUS  = 250
QAM_SIGNAL_RADIUS   = 150

TARGET_RSSI_THRESHOLD = 0.85
MAX_STEPS = 500

TARGET_MODE = 'random'
OBS_MODE    = 'random'

# 매뉴얼 좌표 (BPSK는 상삼각으로 고정해둠)
MANUAL_BPSK_POS  = np.array([150.0, 550.0]) 
MANUAL_QAM0_POS  = np.array([400.0, 200.0]) 
MANUAL_QAM1_POS  = np.array([550.0, 450.0]) 

class DroneEnv:
    def __init__(self):
        self.map_size    = MAP_SIZE
        self.drone_start = START_POS.copy()
        self.state_dim   = 29
        self.action_dim  = 2
        self.obstacles = []
        self.obs_radii = []

    def reset(self):
        self.drone_pos   = self.drone_start.copy()
        self.prev_action = np.array([1.0, 1.0]) / np.sqrt(2)
        self.search_vec  = np.array([1.0, 1.0]) / np.sqrt(2)
        self.steps       = 0
        self.prev_bpsk_rssi = 0.0
        self.prev_qam0_rssi = 0.0
        self.prev_qam1_rssi = 0.0

        self.GRID_N      = 14
        self.visit_count = np.zeros((self.GRID_N, self.GRID_N), dtype=np.int32) 

        # ── [수정] 장애물 생성 (출발지 보호 및 겹침 방지) ──
        self.obstacles = []
        self.obs_radii = []
        
        if OBS_MODE == 'manual':
            # 매뉴얼 모드는 기존 MANUAL_OBSTACLES 사용 (필요 시 정의)
            pass 
        else:
            num_obs = np.random.randint(3, MAX_OBS_COUNT + 1)
            for _ in range(num_obs):
                r = np.random.uniform(OBS_RADIUS_MIN, OBS_RADIUS_MAX)
                
                # 유효한 위치를 찾을 때까지 반복
                for _ in range(100): # 무한 루프 방지
                    px = np.random.uniform(r + 10, self.map_size - r - 10)
                    py = np.random.uniform(r + 10, self.map_size - r - 10)
                    new_pos = np.array([px, py])
                    
                    # 1. 출발지 보호 구역 체크 (100, 100 이내에는 생성 금지)
                    # 장애물의 테두리가 100, 100 선을 침범하지 않도록 r만큼 여유를 둡니다.
                    if px < 100 + r and py < 100 + r:
                        continue
                    
                    # 2. 기존 장애물과 겹치는지 체크
                    collision = False
                    for existing_pos, existing_r in zip(self.obstacles, self.obs_radii):
                        dist = np.linalg.norm(new_pos - existing_pos)
                        # 두 반지름의 합 + 최소 여유 공간(20px) 보다 가까우면 겹친 것으로 간주
                        if dist < (r + existing_r + 20.0):
                            collision = True
                            break
                    
                    if not collision:
                        self.obstacles.append(new_pos)
                        self.obs_radii.append(r)
                        break
            
            # 신경망 입력 차원(29차원) 유지를 위해 빈 공간 채우기
            while len(self.obstacles) < MAX_OBS_COUNT:
                self.obstacles.append(np.array([-1000.0, -1000.0]))
                self.obs_radii.append(0.0)

        # ── 신호원 생성 ──
        if TARGET_MODE == 'manual':
            self.bpsk_pos = MANUAL_BPSK_POS.copy()
            self.qam_pos  = [MANUAL_QAM0_POS.copy(), MANUAL_QAM1_POS.copy()]
        else:
            self.bpsk_pos = self._spawn_signal(
                margin=30, min_gap_start=220, others=[], only_upper=True)
            
            qam0 = self._spawn_signal(
                margin=20, min_gap_start=150,
                others=[self.bpsk_pos], min_gap_others=QAM_SIGNAL_RADIUS + 50, 
                only_upper=False)
            qam1 = self._spawn_signal(
                margin=20, min_gap_start=150,
                others=[self.bpsk_pos, qam0], min_gap_others=QAM_SIGNAL_RADIUS + 50, 
                only_upper=False)
            self.qam_pos = [qam0, qam1]

        return self.get_state()

    def _spawn_signal(self, margin, min_gap_start, others, min_gap_others=0, only_upper=False):
        """
        신호원 생성 함수
        only_upper=True 일 때 '오른쪽 위 삼각형' 구역에만 생성
        """
        while True:
            pos = np.random.uniform(70, self.map_size - 70, size=2)
            
            # [수정] '오른쪽 위 삼각형' (Top-Right Triangle) 조건
            # x + y가 MAP_SIZE보다 커야 대각선 오른쪽 위에 위치하게 됩니다.
            if only_upper:
                if (pos[0] + pos[1]) <= (self.map_size + 50): # 50은 대각선에서 살짝 띄우는 여유
                    continue

            # (이하 동일한 거리 및 충돌 체크 로직...)
            if np.linalg.norm(pos - START_POS) < min_gap_start:
                continue
            
            collision = False
            for obs, r in zip(self.obstacles, self.obs_radii):
                if r == 0: continue
                if np.linalg.norm(pos - obs) < r + margin:
                    collision = True
                    break
            if collision: continue

            if others and min_gap_others > 0:
                if any(np.linalg.norm(pos - o) < min_gap_others for o in others):
                    continue
            
            return pos

    def get_bpsk_rssi(self, pos):
        dist = np.linalg.norm(self.bpsk_pos - pos)
        if dist < BPSK_SIGNAL_RADIUS:
            return 1.0 - dist / BPSK_SIGNAL_RADIUS
        return 0.0

    def get_qam_rssi(self, pos, idx):
        dist = np.linalg.norm(self.qam_pos[idx] - pos)
        if dist < QAM_SIGNAL_RADIUS:
            return 1.0 - dist / QAM_SIGNAL_RADIUS
        return 0.0

    def get_state(self):
        curr_bpsk = self.get_bpsk_rssi(self.drone_pos)
        curr_qam0 = self.get_qam_rssi(self.drone_pos, 0)
        curr_qam1 = self.get_qam_rssi(self.drone_pos, 1)
        bpsk_diff = curr_bpsk - self.prev_bpsk_rssi
        qam0_diff = curr_qam0 - self.prev_qam0_rssi
        qam1_diff = curr_qam1 - self.prev_qam1_rssi

        if curr_bpsk > 0.0:
            vec = self.bpsk_pos - self.drone_pos
            n = np.linalg.norm(vec)
            sensor_dir = vec / n if n > 0.001 else np.zeros(2)
        else:
            sensor_dir = self.search_vec

        state = [
            self.drone_pos[0] / self.map_size,
            self.drone_pos[1] / self.map_size,
            (self.map_size - self.drone_pos[0]) / self.map_size,
            (self.map_size - self.drone_pos[1]) / self.map_size,
            curr_bpsk,
            bpsk_diff * 100.0,
            self.prev_action[0],
            self.prev_action[1],
            sensor_dir[0],
            sensor_dir[1],
            curr_qam0,
            qam0_diff * 100.0,
            curr_qam1,
            qam1_diff * 100.0,
        ]

        for obs, r in zip(self.obstacles, self.obs_radii):
            rx = (obs[0] - self.drone_pos[0]) / self.map_size
            ry = (obs[1] - self.drone_pos[1]) / self.map_size
            dist_to_center = np.linalg.norm([rx, ry])
            dist_to_surface = max(0.0, dist_to_center - (r / self.map_size))
            state.extend([rx, ry, dist_to_surface])

        return np.array(state, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────
    def _apf_repulsion(self, pos, velocity):
        """
        APF 척력 - 접선 편향 방식
        장애물이 드론을 밀어내되, 정면 충돌 시 접선 방향으로 편향시켜
        우회 경로를 유도한다.
        """
        APF_MAX       = 0.85     # 최대 보정 크기 상향

        correction = np.zeros(2)

        for obs, r in zip(self.obstacles, self.obs_radii):
            if r == 0: continue 
            
            d = np.linalg.norm(pos - obs)
            apf_influence = r + 150.0 # 장애물 반경에 맞춰 밀어내는 거리 조절 
            
            if d >= apf_influence or d < 1.0:
                continue

            rep_dir   = (pos - obs) / d
            magnitude = 720000.0 * (1.0/d - 1.0/apf_influence) / (d * d)

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
        WALL_INFLUENCE = 120.0   # 벽 척력 영향 거리 확장
        WALL_K         = 202500.0  # 영향권 1.5배 → 계수 3.4배 스케일
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
        for obs, r in zip(self.obstacles, self.obs_radii):
            if r == 0: continue 
            
            d = np.linalg.norm(pos - obs)
            obs_soft = r + 80.0 
            obs_hard = r + 20.0 

            if obs_hard < d < obs_soft:
                t = (obs_soft - d) / (obs_soft - obs_hard)
                penalty -= t * 200.0
            elif r < d <= obs_hard:
                t = (obs_hard - d) / (obs_hard - r)
                penalty -= t * t * 800.0
        return penalty

    # ──────────────────────────────────────────────────────────────
    def step(self, action):
        move_step = 10.0 # 드론의 최대 이동 거리 (속도) 조절

        action   = np.clip(action, -1.0, 1.0)
        a_norm   = np.linalg.norm(action)
        action   = action / a_norm if a_norm > 0.001 else self.prev_action

        inertia  = 0.3 # 관성 계수 (0.0: 즉시 반응, 1.0: 완전 관성)
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
        for obs, r in zip(self.obstacles, self.obs_radii):
            if r == 0: continue 
            if np.linalg.norm(new_pos - obs) <= r: 
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
            reward = 5000.0  # 이거 많이주면 안되나? 1만점 주고 끝내면 모델이 빨리 목표를 찾는 법을 배우긴 할텐데... 
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
                # 장애물 근처일수록 RSSI 보상 억제 → APF가 작동할 여지 확보
                min_obs_dist_ratio = 1.0
                for obs, r in zip(self.obstacles, self.obs_radii):
                    if r == 0: continue
                    d = np.linalg.norm(self.drone_pos - obs)
                    ratio = (d - r) / (r * 2.0)
                    if ratio < min_obs_dist_ratio:
                        min_obs_dist_ratio = ratio
                        
                obs_suppress = np.clip(min_obs_dist_ratio, 0.3, 1.0)
                rssi_reward = bpsk_diff * 300.0 * obs_suppress

                # 장애물 회피 방향 보상
                obs_avoid_bonus = 0.0
                for obs, r in zip(self.obstacles, self.obs_radii):
                    if r == 0: continue
                    d = np.linalg.norm(self.drone_pos - obs)
                    obs_soft = r + 80.0 
                    if d < obs_soft:
                        away_dir = self.drone_pos - obs
                        away_n   = np.linalg.norm(away_dir)
                        if away_n > 0.001:
                            away_dir /= away_n
                        avoid_align = np.dot(velocity, away_dir)
                        proximity   = (obs_soft - d) / obs_soft
                        obs_avoid_bonus += avoid_align * proximity * 30.0

                reward = (step_penalty
                          + alignment       * 8.0  # AOA 방향 정렬
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
                gx = int(np.clip(new_pos[0] / self.map_size * self.GRID_N, 0, self.GRID_N - 1))
                gy = int(np.clip(new_pos[1] / self.map_size * self.GRID_N, 0, self.GRID_N - 1))

                visit_penalty = -self.visit_count[gx, gy] * 1  # 방문 횟수당 패널티
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