import numpy as np
import torch
import gymnasium as gym
import os
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
    num_envs      = 64   # 한번에 날아가는 드론 수
    steps_per_env = 256  # 드론 수 * 4 로 유지 
    max_updates   = 2000

    # ── 엔트로피 스케줄 파라미터 ────────────────────────────────────
    # 구조: [0~warmup] 고정 → [warmup~max] 코사인 감소
    #
    # 기존 선형 감소의 문제:
    #   1. decay_limit(1000) 이후 500 업데이트가 min_entropy 고정 상태
    #   2. 초반부터 꾸준히 감소해 탐험 초기 보장이 부족
    #
    # 코사인 감소 장점:
    #   - warmup 동안 탐험 충분히 보장
    #   - 이후 완만하게 시작해 후반에 집중 감소 (자연스러운 수렴)
    # ──────────────────────────────────────────────────────────────
    initial_entropy = 0.02
    min_entropy     = 0.005
    warmup_updates  = 300            # 이 구간은 entropy 고정 (탐험 보장)

    envs      = AsyncVectorEnv([make_env for _ in range(num_envs)])
    dummy_env = DroneEnv()
    agent     = PPO(dummy_env.state_dim, dummy_env.action_dim)

#    if os.path.exists('ppo_drone_best.pth'):
#        agent.policy.load_state_dict(torch.load('ppo_drone_best.pth'))
#        print("🚀 [Load] 기존 베스트 모델을 로드했습니다.")

    episode_rewards  = []
    episode_endings  = []
    current_ep_rews  = np.zeros(num_envs)

    print(f"=== 드론 PPO 학습 시작 (Device: {agent.device}) ===")

    states, _ = envs.reset()

    for update in range(1, max_updates + 1):

        # ── 엔트로피 코사인 스케줄 ──────────────────────────────────
        if update <= warmup_updates:
            # 워밍업 구간: 탐험 최대치 고정
            current_entropy = initial_entropy
        else:
            # 워밍업 이후: 코사인 곡선으로 부드럽게 감소
            # progress 0.0 → 1.0 으로 진행될 때
            # cos(0)=1 → cos(π)=-1 이므로 entropy는 initial → min 으로 감소
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
                    current_ep_rews[i] = 0

        agent.update(memory, current_entropy)

        # ── 로그 출력 ────────────────────────────────────────────────
        if update % 10 == 0 and episode_rewards:
            avg_100 = np.mean(episode_rewards[-100:])
            recent_ends = episode_endings[-100:]
            rc = Counter(recent_ends)
            n_r = len(recent_ends)

            succ_pct  = rc.get('Target Found', 0) / n_r * 100
            crash_pct = (rc.get('Obs Crash', 0) + rc.get('Wall Crash', 0) + rc.get('Crash', 0)) / n_r * 100

            # 엔트로피 값도 함께 출력해 스케줄 진행 확인 가능
            print(f"Update {update:4d}/{max_updates} | "
                  f"Eps: {len(episode_rewards):5d} | "
                  f"Avg(100): {avg_100:8.2f} | "
                  f"✅{succ_pct:4.1f}% 💥{crash_pct:4.1f}% | ")

        if update > 50:
            agent.save_if_best(np.mean(episode_rewards[-num_envs:]), 'ppo_drone_best.pth')

    envs.close()
    np.save('rewards_history.npy', np.array(episode_rewards))
    print("\n=== 학습 완료! ===")

if __name__ == '__main__':
    train()