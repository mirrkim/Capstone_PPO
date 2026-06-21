# ── TensorFlow 로그 억제: 반드시 다른 모든 import보다 먼저 실행 ──
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      # ERROR만 표시 (INFO/WARNING 숨김)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'     # "oneDNN custom operations are on" 제거
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'          # absl::InitializeLog 관련 메시지 억제

import numpy as np
import torch
import gymnasium as gym
import math                          # ← 코사인 스케줄용 추가
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
from collections import Counter
from env import DroneEnv
from ppo import PPO

class GymDroneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = DroneEnv()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.env.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.env.action_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.env.reset(), {}

    def step(self, action):
        state, reward, done, info_str = self.env.step(action)
        return state, reward, done, False, {'msg': info_str}

def make_env():
    return GymDroneEnv()

def train():
    num_envs      = 16   # 한번에 날아가는 드론 수 (메모리 한계로 16 고정)
    steps_per_env = 128
    max_updates   = 300
    
    initial_entropy = 0.02
    min_entropy     = 0.005
    warmup_updates  = 100           # 이 구간은 entropy 고정 (탐험 보장)

    envs      = AsyncVectorEnv([make_env for _ in range(num_envs)])
    dummy_env = DroneEnv()
    agent     = PPO(dummy_env.state_dim, dummy_env.action_dim)

    ## 이어서 학습하기
    if os.path.exists('ppo_drone_best.pth'):
        agent.policy.load_state_dict(torch.load('ppo_drone_best.pth'))
        print("🚀 [Load] 기존 베스트 모델을 로드했습니다.")
  
    episode_rewards   = []
    episode_endings   = []
    
    # 🚀 [추가] 실시간 성공 여부(1 또는 0)와 최근 100개 기준 성공률 추이를 저장할 리스트
    success_history   = [] 
    accuracy_over_time = [] 
    
    current_ep_rews   = np.zeros(num_envs)

    print(f"=== 드론 PPO 학습 시작 (Device: {agent.device}) ===")

    states, _ = envs.reset()

    for update in range(1, max_updates + 1):

        # ── 엔트로피 코사인 스케줄 ──────────────────────────────────
        if update <= warmup_updates:
            current_entropy = initial_entropy
        else:
            progress = (update - warmup_updates) / (max_updates - warmup_updates)
            current_entropy = min_entropy + 0.5 * (initial_entropy - min_entropy) \
                              * (1 + math.cos(math.pi * progress))
        # ──────────────────────────────────────────────────────────

        memory = {k: [] for k in ('states', 'actions', 'log_probs', 'rewards', 'dones', 'values')}

        for _ in range(steps_per_env):
            actions, log_probs, values = agent.select_action(states)
            next_states, rewards, terminations, truncations, infos = envs.step(actions)
            dones = np.logical_or(terminations, truncations)

            memory['states'].append(states)
            memory['actions'].append(actions)
            memory['log_probs'].append(log_probs)
            memory['rewards'].append(rewards)
            memory['dones'].append(dones)
            memory['values'].append(values)

            states = next_states
            current_ep_rews += rewards

            for i, done in enumerate(dones):
                if done:
                    # ── 종료 원인 추출 ──────────────────────────────
                    msg = None
                    if 'final_info' in infos and infos['final_info'][i] is not None:
                        msg = infos['final_info'][i].get('msg', None)
                    if msg is None and 'msg' in infos:
                        msg = infos['msg'][i]
                    if msg is None:
                        msg = 'Target Found' if current_ep_rews[i] > 3000 else 'Crash'
                    # ───────────────────────────────────────────────

                    episode_rewards.append(current_ep_rews[i])
                    episode_endings.append(str(msg))
                    
                    # 🚀 [추가] 성공하면 1, 실패(충돌/타임아웃)하면 0을 기록합니다.
                    if str(msg) == 'Target Found':
                        success_history.append(1)
                    else:
                        success_history.append(0)
                        
                    current_ep_rews[i] = 0

        agent.update(memory, current_entropy)

        # 🚀 [추가] 매 업데이트가 끝날 때마다 '최근 100개 에피소드'의 정확도 평균을 계산하여 저장
        if len(success_history) > 0:
            # 에피소드가 아직 100개 미만일 때는 전체 평균을, 100개 이상일 때는 최근 100개의 평균을 구합니다.
            recent_success = success_history[-100:]
            current_accuracy = np.mean(recent_success) * 100  # 퍼센트(%) 단위 변환
            accuracy_over_time.append(current_accuracy)
        else:
            accuracy_over_time.append(0.0)

        # ── 로그 출력 ────────────────────────────────────────────────
        if update % 10 == 0 and episode_rewards:
            avg_100 = np.mean(episode_rewards[-100:])
            recent_ends = episode_endings[-100:]
            rc = Counter(recent_ends)
            n_r = len(recent_ends)

            succ_pct  = rc.get('Target Found', 0) / n_r * 100
            crash_pct = (rc.get('Obs Crash', 0) + rc.get('Wall Crash', 0) + rc.get('Crash', 0)) / n_r * 100

            print(f"Update {update:4d}/{max_updates} | "
                  f"Eps: {len(episode_rewards):5d} | "
                  f"Avg(100): {avg_100:8.2f} | "
                  f"✅{succ_pct:4.1f}% 💥{crash_pct:4.1f}% | ")

        if update > 50:
            agent.save_if_best(np.mean(episode_rewards[-num_envs:]), 'ppo_drone_best.pth')

    envs.close()
    
    # 💾 데이터 저장 파트
    np.save('rewards_history.npy', np.array(episode_rewards))
    # 🚀 [추가] 정확도(성공률) 변화 히스토리를 넘파이 파일로 저장합니다.
    np.save('accuracy_history.npy', np.array(accuracy_over_time))
    
    print("\n=== 학습 완료 및 데이터 저장 완료 (`rewards_history.npy`, `accuracy_history.npy`) ===")

if __name__ == '__main__':
    train()