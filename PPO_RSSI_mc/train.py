import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
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
        info = {'msg': info_str}
        return state, reward, done, False, info

def make_env():
    return GymDroneEnv()

def train():
    num_envs = 32
    envs = AsyncVectorEnv([make_env for _ in range(num_envs)])
    
    dummy_env = DroneEnv()
    agent = PPO(dummy_env.state_dim, dummy_env.action_dim)

    steps_per_env = 128
    max_updates   = 500  # 업데이트 10회 당 약 300 에피소드
    
    episode_rewards = []
    current_ep_rewards = np.zeros(num_envs)

    print("=== 드론 PPO 멀티코어 학습 시작 (오른쪽 위 타겟 집중 공략) ===")
    print(f"    동시 출격 드론 수 : {num_envs}대\n")

    states, _ = envs.reset()

    for update in range(1, max_updates + 1):
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
            current_ep_rewards += rewards 

            for i, done in enumerate(dones):
                if done:
                    episode_rewards.append(current_ep_rewards[i])
                    current_ep_rewards[i] = 0

        # PPO 업데이트
        agent.update(memory)
        # [수정] 이번 업데이트에서 수집된 모든 보상의 평균 계산
        # memory['rewards']는 (steps_per_env, num_envs) 형태의 리스트입니다.
        if update > (max_updates - 50): 
            current_update_reward = np.mean(memory['rewards'])
            agent.save_if_best(current_update_reward, 'ppo_drone.pth')

        # 진행 상황만 간단히 출력
        if update % 10 == 0 and len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Update: {update:4d}/{max_updates} | Episodes: {len(episode_rewards):5d} | Avg Reward: {avg_reward:8.2f}")

    envs.close() 

    # ── [핵심] 학습이 완전히 끝난 후 딱 한 번 저장 ──
    torch.save(agent.policy.state_dict(), 'ppo_drone.pth')
    np.save('rewards_history.npy', np.array(episode_rewards))
    
    print("\n=== 학습 완료! 최종 가중치가 'ppo_drone.pth'에 반영되었습니다. ===")

if __name__ == '__main__':
    train()