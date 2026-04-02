import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import gymnasium as gym
from gymnasium import spaces

from env import (DroneEnv, MAP_SIZE, 
                 BPSK_SIGNAL_RADIUS, QAM_SIGNAL_RADIUS,
                 TARGET_MODE)
from ppo import PPO

# --- Gymnasium 래퍼 클래스 ---
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

def evaluate_and_animate():
    """
    최종 학습된 'ppo_drone.pth' 가중치를 사용하여 테스트 비행을 수행하고 애니메이션을 출력합니다.
    """
    gym_env = GymDroneEnv()
    env = gym_env.env 
    agent = PPO(env.state_dim, env.action_dim)

    # ── 모델 / 학습 기록 불러오기 ─────────────────────────────────
    try:
        # [수정] train.py에서 저장한 최종 모델 파일명으로 변경
        agent.policy.load_state_dict(
            torch.load('ppo_drone.pth', map_location='cpu'))
        rewards_history = np.load('rewards_history.npy')
    except FileNotFoundError:
        print("에러: 'ppo_drone.pth' 또는 'rewards_history.npy' 파일이 없습니다.")
        print("먼저 train.py를 실행하여 학습을 완료하세요.")
        return

    print(f"=== 테스트 시작 (신호원 배치 모드: {TARGET_MODE}) ===")

    # ── 시뮬레이션 실행 ──────────────────────────────────────────
    state, _ = gym_env.reset()
    trajectory = [env.drone_pos.copy()]
    device = next(agent.policy.parameters()).device

    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) 
        with torch.no_grad():
            mu = agent.policy.actor(state_tensor)
        action = mu.cpu().numpy()[0] # 결정론적 행동 (평균값 사용)

        next_state, reward, terminated, truncated, info = gym_env.step(action)
        done = terminated or truncated
        state = next_state
        trajectory.append(env.drone_pos.copy())

        if done:
            print(f"비행 종료: {info['msg']} (총 {len(trajectory)} 스텝)")
            break

    trajectory = np.array(trajectory)

    # ── 화면 구성 ─────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("PPO Drone – BPSK Signal Tracking (Final Test)", fontsize=14, fontweight='bold')

    # [왼쪽] 비행 애니메이션 레이아웃
    ax1.set_xlim(0, MAP_SIZE)
    ax1.set_ylim(0, MAP_SIZE)
    ax1.set_title("Flight Animation")
    ax1.set_aspect('equal')
    ax1.set_facecolor('#f5f5f5')

    # BPSK 타겟 (빨간 원)
    bpsk_circle = plt.Circle(env.bpsk_pos, BPSK_SIGNAL_RADIUS, color='red', alpha=0.15, zorder=1)
    ax1.add_patch(bpsk_circle)
    ax1.plot(*env.bpsk_pos, 'ro', markersize=7, zorder=5, label='BPSK (Target)')

    # QAM 방해꾼 (파란 네모)
    for i, qpos in enumerate(env.qam_pos):
        qam_circle = plt.Circle(qpos, QAM_SIGNAL_RADIUS, color='blue', alpha=0.12, zorder=1)
        ax1.add_patch(qam_circle)
        label = '64QAM (Ignored)' if i == 0 else '_nolegend_'
        ax1.plot(*qpos, 'bs', markersize=7, zorder=5, label=label)

    # [수정] 동적 장애물 (노란 원) - 실제 생성된 반지름 사용
    real_obs_count = 0
    for obs, r in zip(env.obstacles, env.obs_radii):
        if r <= 0: continue # 가짜 장애물 패스
        real_obs_count += 1
        obs_circle = plt.Circle(obs, r, color='#FFD700', ec='black', linewidth=1.2, zorder=3)
        ax1.add_patch(obs_circle)

    ax1.plot(*env.drone_start, 'g^', markersize=9, zorder=6, label='Start')
    drone_dot, = ax1.plot([], [], 'o', color='purple', markersize=9, zorder=7, label='Drone')
    path_line, = ax1.plot([], [], '-', color='purple', alpha=0.45, zorder=6)

    # 범례 설정
    obs_patch = mpatches.Patch(color='#FFD700', label=f'Obstacle ×{real_obs_count}')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles + [obs_patch], labels + [obs_patch.get_label()], loc='upper right', fontsize=8)

    # [오른쪽] 학습 그래프 레이아웃
    ax2.plot(rewards_history, color='gray', alpha=0.3, linewidth=0.8, label='Episode Reward')
    if len(rewards_history) >= 100:
        moving_avg = np.convolve(rewards_history, np.ones(100)/100, mode='valid')
        ax2.plot(range(99, len(rewards_history)), moving_avg, color='darkorange', linewidth=1.5, label='Moving Avg (100)')
    ax2.set_title("Training Reward History")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)

    # ── 애니메이션 실행 ──────────────────────────────────────────
    def update(frame):
        drone_dot.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
        path_line.set_data(trajectory[:frame + 1, 0], trajectory[:frame + 1, 1])
        return drone_dot, path_line

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=40, blit=True, repeat=False)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    evaluate_and_animate()