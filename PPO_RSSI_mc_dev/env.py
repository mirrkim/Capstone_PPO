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

class DroneEnv:
    def __init__(self):
        self.map_size    = MAP_SIZE
        self.drone_start = START_POS.copy()
        self.state_dim   = 30 
        self.action_dim  = 2
        self.MACRO_N     = 2
        self.obstacles   = []
        self.obs_radii   = []
        self.qam_pos     = [np.array([400, 200]), np.array([200, 400])]

        self.macro_centers = [
            np.array([175.0, 175.0]), np.array([525.0, 175.0]),
            np.array([175.0, 525.0]), np.array([525.0, 525.0])
        ]

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
        
        # [추가] 갈대 같은 마음을 다잡을 타겟 메모리 
        self.chosen_target = -1 
        
        self._generate_world()
        return self.get_state()

    def _generate_world(self):
        self.obstacles = []; self.obs_radii = []
        num_obs = np.random.randint(MIN_OBS_COUNT, MAX_OBS_COUNT + 1)
        for _ in range(num_obs):
            r = np.random.uniform(OBS_RADIUS_MIN, OBS_RADIUS_MAX)
            px = np.random.uniform(r + 10, self.map_size - r - 10)
            py = np.random.uniform(r + 10, self.map_size - r - 10)
            if px < 180 and py < 180: continue
            self.obstacles.append(np.array([px, py])); self.obs_radii.append(r)
        
        while len(self.obstacles) < MAX_OBS_COUNT:
            self.obstacles.append(np.array([-1000.0, -1000.0])); self.obs_radii.append(0.0)
            
        while True:
            px, py = np.random.uniform(50, 650), np.random.uniform(50, 650)
            if px + py > 700:
                candidate_pos = np.array([px, py])
                is_safe = True
                for obs, r in zip(self.obstacles, self.obs_radii):
                    if r > 0 and np.linalg.norm(candidate_pos - obs) < (r + 30.0):
                        is_safe = False; break
                if is_safe:
                    self.bpsk_pos = candidate_pos; break

    def get_bpsk_rssi(self, pos):
        dist = np.linalg.norm(self.bpsk_pos - pos)
        return max(0.0, 1.0 - dist / BPSK_SIGNAL_RADIUS) if dist < BPSK_SIGNAL_RADIUS else 0.0

    def get_state(self):
        curr_bpsk = self.get_bpsk_rssi(self.drone_pos)
        sensor_dir = (self.bpsk_pos - self.drone_pos) / (np.linalg.norm(self.bpsk_pos - self.drone_pos) + 1e-8) if curr_bpsk > 0.0 else self.search_vec

        qam_aoas = []
        for qpos in self.qam_pos:
            dist = np.linalg.norm(qpos - self.drone_pos)
            if dist < QAM_SIGNAL_RADIUS:
                q_dir = (qpos - self.drone_pos) / (dist + 1e-8)
                qam_aoas.extend([q_dir[0], q_dir[1]])
            else:
                qam_aoas.extend([0.0, 0.0])

        state = [
            self.drone_pos[0]/700, self.drone_pos[1]/700,
            (700 - self.drone_pos[0])/700, (700 - self.drone_pos[1])/700,
            curr_bpsk, (curr_bpsk - self.prev_bpsk_rssi) * 100,
            self.prev_action[0], self.prev_action[1],
            sensor_dir[0], sensor_dir[1],
            qam_aoas[0], qam_aoas[1], qam_aoas[2], qam_aoas[3],
            self.belief[0], self.belief[1], self.belief[2], self.belief[3]
        ]

        obs_info = []
        for obs, r in zip(self.obstacles, self.obs_radii):
            if r <= 0: continue 
            rel_pos = (obs - self.drone_pos) / 700
            dist = np.linalg.norm(rel_pos) - (r / 700)
            obs_info.append((dist, rel_pos[0], rel_pos[1], max(0, dist)))
        
        obs_info.sort(key=lambda x: x[0])
        for i in range(4):
            if i < len(obs_info):
                info = obs_info[i]; state.extend([info[1], info[2], info[3]])
            else:
                state.extend([0.0, 0.0, 0.0])
        return np.array(state, dtype=np.float32)

    def step(self, action):
        done = False
        move_step = 10.0
        action = np.clip(action, -1.0, 1.0)
        
        pre_min_dist = 999.0
        for obs, r in zip(self.obstacles, self.obs_radii):
            if r > 0:
                d = np.linalg.norm(self.drone_pos - obs) - r
                if d < pre_min_dist: pre_min_dist = d
        
        pre_wall_dist = min(self.drone_pos[0], self.drone_pos[1], 700 - self.drone_pos[0], 700 - self.drone_pos[1])
        danger_level = max(1.0 - np.clip(pre_min_dist / 35.0, 0.0, 1.0), 1.0 - np.clip(pre_wall_dist / 40.0, 0.0, 1.0))
        alpha = 0.40 - 0.15 * danger_level
        
        velocity = action * alpha + self.prev_action * (1.0 - alpha)
        v_norm = np.linalg.norm(velocity)
        velocity = velocity / v_norm if v_norm > 0.001 else self.prev_action
        
        self.prev_action = velocity
        new_pos = self.drone_pos + velocity * move_step

        if new_pos[0] < 0 or new_pos[0] > 700 or new_pos[1] < 0 or new_pos[1] > 700:
            return self.get_state(), -2000.0, True, "Wall Crash"
        for obs, r in zip(self.obstacles, self.obs_radii):
            if r > 0 and np.linalg.norm(new_pos - obs) <= r:
                return self.get_state(), -2000.0, True, "Obs Crash"

        obs_margin, wall_margin = 70.0, 70.0
        near_crash_margin = 20.0          

        min_dist_to_obs = 999.0
        for obs, r in zip(self.obstacles, self.obs_radii):
            if r > 0:
                d = np.linalg.norm(new_pos - obs) - r
                if d < min_dist_to_obs: min_dist_to_obs = d
        
        soft_penalty = 0.0
        if min_dist_to_obs < obs_margin:
            soft_penalty -= ((obs_margin - min_dist_to_obs) / obs_margin) ** 2 * 150.0
        if min_dist_to_obs < near_crash_margin:
            soft_penalty -= 300.0
        
        wall_dist = min(new_pos[0], new_pos[1], 700 - new_pos[0], 700 - new_pos[1])
        if wall_dist < wall_margin:
            soft_penalty -= ((wall_margin - wall_dist) / wall_margin) ** 2 * 130.0

        self.drone_pos, self.steps = new_pos, self.steps + 1
        curr_bpsk = self.get_bpsk_rssi(new_pos)
        bpsk_diff = curr_bpsk - self.prev_bpsk_rssi

        quad_idx = int(np.clip(new_pos[1] / 350.0, 0, 1)) * 2 + int(np.clip(new_pos[0] / 350.0, 0, 1))
        macro_crossing_bonus = 0.0
        if quad_idx != self.current_macro:
            if quad_idx not in self.visited_quad_history:
                macro_crossing_bonus = 300.0
            self.visited_quad_history.append(self.current_macro)
            self.current_macro, self.macro_step_count = quad_idx, 0
            self.macro_entry_freeze = BELIEF_FREEZE_STEPS
        else:
            self.macro_step_count += 1
            if self.macro_entry_freeze > 0:
                self.macro_entry_freeze -= 1

        if curr_bpsk > 0.0:
            self.belief *= 0.0; self.belief[quad_idx] = 1.0
        else:
            if self.macro_entry_freeze == 0:
                self.belief[quad_idx] *= BELIEF_DECAY
                self.belief /= np.sum(self.belief)

        curr_entropy = -np.sum(self.belief * np.log(self.belief + 1e-8))
        info_reward = (self.prev_entropy - curr_entropy) * 10.0
        self.prev_entropy = curr_entropy

        if curr_bpsk >= TARGET_RSSI_THRESHOLD:
            return self.get_state(), 5000.0, True, "Target Found"

        if curr_bpsk > 0.0:
            step_penalty = -0.8
            rssi_diff_rew = bpsk_diff * 300.0
            proximity_reward = curr_bpsk * 80.0

            a, b = new_pos, self.bpsk_pos
            ab = b - a
            len_sq = np.dot(ab, ab)
            path_blocked = False
            if len_sq > 0.001:
                for obs, r in zip(self.obstacles, self.obs_radii):
                    if r <= 0: continue
                    t = np.clip(np.dot(obs - a, ab) / len_sq, 0.0, 1.0)
                    if np.linalg.norm(obs - (a + t * ab)) < (r + 20.0):
                        path_blocked = True; break
            if path_blocked:
                proximity_reward *= 0.2

            reward = step_penalty + rssi_diff_rew + proximity_reward + soft_penalty + info_reward
        else:
            step_penalty = -0.8
            self.search_vec = velocity.copy()

            temp_belief = self.belief.copy(); temp_belief[quad_idx] = -1.0
            max_p = np.max(temp_belief)
            candidates = np.where(np.isclose(temp_belief, max_p))[0]
            
            # ─── [핵심 수정] 타겟 고착(Sticky Target) 로직 ───
            # 기존에 골랐던 타겟이 여전히 최고 확률 후보군(candidates)에 있다면?
            # 거리 따위 무시하고 줏대 있게 밀어붙입니다!
            if self.chosen_target in candidates:
                best_target = self.chosen_target
            else:
                # 후보가 여러 개인데 기존 타겟이 죽었다면 새로 하나 뽑습니다.
                if len(candidates) > 1:
                    min_cost = float('inf')
                    best_target = candidates[0]
                    for c in candidates:
                        target_center = self.macro_centers[c]
                        dist_cost = np.linalg.norm(target_center - self.drone_pos)
                        danger_cost, A, B = 0.0, self.drone_pos, target_center
                        AB = B - A; len_sq = np.dot(AB, AB)
                        if len_sq > 0.001:
                            for obs, r in zip(self.obstacles, self.obs_radii):
                                if r > 0:
                                    t = np.clip(np.dot(obs - A, AB) / len_sq, 0.0, 1.0)
                                    dist_to_path = np.linalg.norm(obs - (A + t * AB))
                                    if dist_to_path < (r + 15.0): danger_cost += r
                        total_cost = dist_cost + (danger_cost * 3.0)
                        if total_cost < min_cost: min_cost, best_target = total_cost, c
                else:
                    best_target = candidates[0]
                
                # 새 타겟을 마음에 새깁니다.
                self.chosen_target = best_target
            # ───────────────────────────────────────────────

            target_pos = self.macro_centers[best_target]
            dir_to_target = (target_pos - self.drone_pos) / (np.linalg.norm(target_pos - self.drone_pos) + 1e-8)
            danger_factor = np.clip(min_dist_to_obs / obs_margin, 0.0, 1.0)

            belief_diff = float(np.clip(self.belief[best_target] - self.belief[quad_idx], 0.0, 0.5))
            multiplier = 30.0 + belief_diff * 180.0   
            drive_reward = np.dot(velocity, dir_to_target) * multiplier * danger_factor

            reward = step_penalty + drive_reward + macro_crossing_bonus + info_reward + soft_penalty

        self.prev_bpsk_rssi = curr_bpsk
        return self.get_state(), reward, self.steps >= MAX_STEPS, "Normal"