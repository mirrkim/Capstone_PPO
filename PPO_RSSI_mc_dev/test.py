import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import gymnasium as gym
from gymnasium import spaces
import os

# env.py에서 설정한 상수들을 그대로 가져옵니다.
from env import (DroneEnv, MAP_SIZE, 
                 BPSK_SIGNAL_RADIUS, QAM_SIGNAL_RADIUS,
                 TARGET_MODE)
from ppo import PPO

# --- Gymnasium 래퍼 클래스 (train.py와 동일하게 유지) ---
class GymDroneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = DroneEnv()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.env.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.env.action_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 여기서 env.reset()이 호출되며 '장기 기억 감쇠(Decay)'가 실행됩니다.
        return self.env.reset(), {}

    def step(self, action):
        state, reward, done, info_str = self.env.step(action)
        info = {'msg': info_str}
        return state, reward, done, False, info

def evaluate_and_animate():
    gym_env = GymDroneEnv()
    env = gym_env.env 
    agent = PPO(env.state_dim, env.action_dim)

    # ── [핵심 수정 1] train.py에서 저장한 파일명에 맞춤 ────────────────
    # 가장 똑똑한 모델(best)을 먼저 찾고, 없으면 최종 모델(final)을 가져옵니다.
    best_path = 'ppo_drone_best.pth'
    final_path = 'ppo_drone_final.pth'
    load_path = best_path if os.path.exists(best_path) else final_path

    try:
        agent.policy.load_state_dict(torch.load(load_path, map_location='cpu'))
        rewards_history = np.load('rewards_history.npy')
        print(f"── 성공: '{load_path}' 가중치를 로드했습니다. ──")
    except FileNotFoundError:
        print(f"에러: 모델 파일이 없습니다. train.py를 실행하여 학습을 먼저 완료하세요.")
        return

    print(f"=== 테스트 비행 시작 (타겟 모드: {TARGET_MODE}) ===")

    # ── 시뮬레이션 실행 (결정론적 비행) ──────────────────────────
    state, _ = gym_env.reset()
    trajectory = [env.drone_pos.copy()]
    device = next(agent.policy.parameters()).device

    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) 
        with torch.no_grad():
            mu = agent.policy.actor(state_tensor)
        action = mu.cpu().numpy()[0] # 테스트 시에는 확률적 샘플링 대신 평균값(mu) 사용

        next_state, reward, terminated, truncated, info = gym_env.step(action)
        done = terminated or truncated
        state = next_state
        trajectory.append(env.drone_pos.copy())

        if done:
            print(f"비행 종료: {info['msg']} (총 {len(trajectory)} 스텝)")
            break

    trajectory = np.array(trajectory)

    # ── [핵심 수정 2] 시각화 레이아웃 최적화 ──────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Drone Mission Test: {load_path}", fontsize=14, fontweight='bold')

    # [왼쪽] 비행 애니메이션 (실제 장애물 크기 반영)
    ax1.set_xlim(0, MAP_SIZE)
    ax1.set_ylim(0, MAP_SIZE)
    ax1.set_aspect('equal')
    ax1.set_facecolor('#fdfdfd')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # BPSK 타겟 영역
    ax1.add_patch(plt.Circle(env.bpsk_pos, BPSK_SIGNAL_RADIUS, color='red', alpha=0.1))
    ax1.plot(*env.bpsk_pos, 'r*', markersize=12, label='BPSK (Target)')

    # QAM 방해꾼 영역
    for i, qpos in enumerate(env.qam_pos):
        ax1.add_patch(plt.Circle(qpos, QAM_SIGNAL_RADIUS, color='blue', alpha=0.08))
        ax1.plot(*qpos, 'bx', markersize=8, label='QAM (Jammer)' if i==0 else "")

    # 장애물 시각화 (env.py에서 생성된 실제 반지름 r 사용)
    for obs, r in zip(env.obstacles, env.obs_radii):
        if r > 0:
            ax1.add_patch(plt.Circle(obs, r, color='#333333', alpha=0.6))

    ax1.plot(*env.drone_start, 'go', markersize=10, label='Start')
    drone_dot, = ax1.plot([], [], 'o', color='magenta', markersize=8, zorder=10)
    path_line, = ax1.plot([], [], '-', color='magenta', alpha=0.5, linewidth=2)
    ax1.legend(loc='upper right', fontsize=8, ncol=2)

    # [오른쪽] 학습 곡선 (rewards_history.npy 시각화)
    ax2.plot(rewards_history, color='lightgray', alpha=0.5, label='Raw Reward')
    if len(rewards_history) >= 50:
        ma = np.convolve(rewards_history, np.ones(50)/50, mode='valid')
        ax2.plot(range(49, len(rewards_history)), ma, color='crimson', linewidth=2, label='Moving Avg(50)')
    ax2.set_title("Training Reward History")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.legend()

    # ── 애니메이션 함수 ──────────────────────────────────────────
    def update(frame):
        drone_dot.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
        path_line.set_data(trajectory[:frame + 1, 0], trajectory[:frame + 1, 1])
        return drone_dot, path_line

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=30, blit=True, repeat=False)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    evaluate_and_animate()