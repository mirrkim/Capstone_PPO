import numpy as np
import torch
from env import DroneEnv
from ppo import PPO


def train():
    """
    PPO 학습 루프

    흐름
    ----
    1. 에피소드마다 env.reset() → 신호원 랜덤 배치
    2. select_action()으로 행동 샘플링 → env.step()으로 전이
    3. experience를 memory에 누적
    4. timestep이 update_timestep(2000)에 도달하면 PPO 업데이트
    5. 100 에피소드마다 평균 보상 출력
    6. 역대 최고 평균 보상 갱신 시 best_ppo_drone.pth 저장
    7. 학습 종료 후 rewards_history.npy 저장
    """
    env   = DroneEnv()
    agent = PPO(env.state_dim, env.action_dim)

    max_episodes    = 5000      # 맵·신호원이 늘었으므로 에피소드 증가
    update_timestep = 2000      # 2000 스텝마다 정책 업데이트

    timestep        = 0
    episode_rewards = []
    best_reward     = -np.inf

    memory = {k: [] for k in ('states', 'actions', 'log_probs',
                               'rewards', 'dones', 'values')}

    print("=== 드론 자율비행 PPO 학습 시작 (BPSK 신호원 탐색) ===")
    print(f"    State dim  : {env.state_dim}")
    print(f"    Action dim : {env.action_dim}")
    print(f"    Map size   : {env.map_size} × {env.map_size}")
    print(f"    Obstacles  : {len(env.obstacles)}개\n")

    for ep in range(1, max_episodes + 1):
        state     = env.reset()
        ep_reward = 0.0
        info      = ""

        while True:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            memory['states'].append(state)
            memory['actions'].append(action)
            memory['log_probs'].append(log_prob)
            memory['rewards'].append(reward)
            memory['dones'].append(done)
            memory['values'].append(value)

            state      = next_state
            ep_reward += reward
            timestep  += 1

            # 일정 timestep마다 정책 업데이트 후 메모리 초기화
            if timestep % update_timestep == 0:
                agent.update(memory)
                memory = {k: [] for k in memory.keys()}

            if done:
                break

        episode_rewards.append(ep_reward)

        if ep % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode: {ep:4d} | "
                  f"Avg Reward (Last 100): {avg_reward:8.2f} | "
                  f"Last Result: {info}")

            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent.policy.state_dict(), 'best_ppo_drone.pth')
                print(f"           ✅ 최고 성능 모델 저장 (avg={best_reward:.2f})")

    np.save('rewards_history.npy', np.array(episode_rewards))
    print("\n=== 학습 완료 | best_ppo_drone.pth / rewards_history.npy 생성됨 ===")


if __name__ == '__main__':
    train()
