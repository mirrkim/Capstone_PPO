import numpy as np
import torch
import gymnasium as gym
import os  # 파일 존재 확인용 추가
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
    num_envs      = 32
    steps_per_env = 128
    max_updates   = 1500
    initial_entropy = 0.02
    min_entropy     = 0.005
    decay_limit     = 1000
    
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
        # 엔트로피 계산은 하되 프린트에서는 제외합니다.
        current_entropy = max(min_entropy, initial_entropy - (initial_entropy - min_entropy) * (update / decay_limit))
        
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
                    # ── [중요] 종료 원인 추출 무적 로직 (0.0% 방지) ──────
                    msg = None
                    if 'final_info' in infos and infos['final_info'][i] is not None:
                        msg = infos['final_info'][i].get('msg', None)
                    if msg is None and 'msg' in infos:
                        msg = infos['msg'][i]
                    
                    # 그래도 없으면 보상 수치로 강제 판단 (점수가 높으면 성공)
                    if msg is None:
                        msg = 'Target Found' if current_ep_rews[i] > 3000 else 'Crash'
                    # ──────────────────────────────────────────────────

                    episode_rewards.append(current_ep_rews[i])
                    episode_endings.append(str(msg))
                    current_ep_rews[i] = 0

        agent.update(memory, current_entropy) 

        # 로그 출력 
        if update % 10 == 0 and episode_rewards:
            avg_100 = np.mean(episode_rewards[-100:])
            recent_ends = episode_endings[-100:]
            rc = Counter(recent_ends)
            n_r = len(recent_ends)
            
            succ_pct  = rc.get('Target Found', 0) / n_r * 100
            crash_pct = (rc.get('Obs Crash', 0) + rc.get('Wall Crash', 0) + rc.get('Crash', 0)) / n_r * 100
            
            # 성공률과 충돌률만 깔끔하게 표시
            print(f"Update {update:4d}/{max_updates} | "
                  f"Eps: {len(episode_rewards):5d} | "
                  f"Avg(100): {avg_100:8.2f} | "
                  f"✅{succ_pct:4.1f}% 💥{crash_pct:4.1f}%")

        if update > 50:
            agent.save_if_best(np.mean(episode_rewards[-num_envs:]), 'ppo_drone_best.pth')

    envs.close()
    print("\n=== 학습 완료! ===")

if __name__ == '__main__':
    train()