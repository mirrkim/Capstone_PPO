import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from env import (DroneEnv, MAP_SIZE, OBS_RADIUS,
                 BPSK_SIGNAL_RADIUS, QAM_SIGNAL_RADIUS,
                 TARGET_MODE)
from ppo import PPO


def evaluate_and_animate():
    """
    학습된 모델을 불러와 시뮬레이션 1회 실행 후 애니메이션을 생성한다.

    화면 구성
    ---------
    왼쪽 : 드론 비행 애니메이션
           - 빨간 반투명 원 : BPSK 신호 반경 (250)
           - 파란 반투명 원 : 64QAM 신호 반경 (150) × 2
           - 노란 원        : 장애물 (반지름 40) × 5
           - 보라색 점/선   : 드론 현재 위치 + 이동 경로
    오른쪽: 에피소드별 학습 보상 그래프

    신호원 배치 모드는 env.py 상단의 TARGET_MODE 상수로 결정된다.
    - 'random' : 랜덤 배치
    - 'manual' : MANUAL_BPSK_POS / MANUAL_QAM0_POS / MANUAL_QAM1_POS 사용
    """
    env   = DroneEnv()
    agent = PPO(env.state_dim, env.action_dim)

    # ── 모델 / 학습 기록 불러오기 ─────────────────────────────────
    try:
        agent.policy.load_state_dict(
            torch.load('best_ppo_drone.pth', map_location='cpu'))
        rewards_history = np.load('rewards_history.npy')
    except FileNotFoundError:
        print("에러: 'best_ppo_drone.pth' 또는 'rewards_history.npy' 없음."
              " 먼저 train.py를 실행하세요.")
        return

    print(f"=== 테스트 시작 (신호원 배치 모드: {TARGET_MODE}) ===")

    # ── 시뮬레이션 실행 (노이즈 없이 actor 평균값 사용) ───────────
    state      = env.reset()
    trajectory = [env.drone_pos.copy()]

    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mu = agent.policy.actor(state_tensor)
        action = mu.numpy()[0]          # 결정론적 행동 (탐험 없음)

        state, reward, done, info = env.step(action)
        trajectory.append(env.drone_pos.copy())

        if done:
            print(f"비행 종료: {info}  (총 {len(trajectory)} 스텝)")
            break

    trajectory = np.array(trajectory)

    # ── 화면 구성 ─────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("PPO Drone – BPSK Signal Tracking", fontsize=14, fontweight='bold')

    # ────────────────────────────────
    # [왼쪽] 애니메이션 환경
    # ────────────────────────────────
    ax1.set_xlim(0, MAP_SIZE)
    ax1.set_ylim(0, MAP_SIZE)
    ax1.set_title("Flight Animation")
    ax1.set_aspect('equal')
    ax1.set_facecolor('#f5f5f5')

    # BPSK 신호 반경 (반투명 빨강)
    bpsk_circle = plt.Circle(env.bpsk_pos, BPSK_SIGNAL_RADIUS,
                              color='red', alpha=0.15, zorder=1)
    ax1.add_patch(bpsk_circle)
    ax1.plot(*env.bpsk_pos, 'ro', markersize=7, zorder=5, label='BPSK (Target)')

    # 64QAM 신호 반경 (반투명 파랑) × 2
    for i, qpos in enumerate(env.qam_pos):
        qam_circle = plt.Circle(qpos, QAM_SIGNAL_RADIUS,
                                color='blue', alpha=0.12, zorder=1)
        ax1.add_patch(qam_circle)
        label = '64QAM (Ignored)' if i == 0 else '_nolegend_'
        ax1.plot(*qpos, 'bs', markersize=7, zorder=5, label=label)

    # 장애물 (노란 원, 검은 테두리)
    for obs in env.obstacles:
        obs_circle = plt.Circle(obs, OBS_RADIUS,
                                color='#FFD700', ec='black', linewidth=1.2, zorder=3)
        ax1.add_patch(obs_circle)

    # 드론 시작 위치 표시
    ax1.plot(*env.drone_start, 'g^', markersize=9, zorder=6, label='Start')

    # 드론 (애니메이션 객체)
    drone_dot, = ax1.plot([], [], 'o', color='purple',
                          markersize=9, zorder=7, label='Drone')
    path_line, = ax1.plot([], [], '-', color='purple', alpha=0.45, zorder=6)

    ax1.legend(loc='upper right', fontsize=8)

    # 장애물 범례 수동 추가
    obs_patch = mpatches.Patch(color='#FFD700', label=f'Obstacle ×{len(env.obstacles)}')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles + [obs_patch], labels + [obs_patch.get_label()],
               loc='upper right', fontsize=8)

    # ────────────────────────────────
    # [오른쪽] 학습 보상 그래프
    # ────────────────────────────────
    ax2.plot(rewards_history, color='gray', alpha=0.5, linewidth=0.8, label='Episode Reward')

    # 100 에피소드 이동 평균
    if len(rewards_history) >= 100:
        moving_avg = np.convolve(rewards_history,
                                 np.ones(100) / 100, mode='valid')
        ax2.plot(range(99, len(rewards_history)), moving_avg,
                 color='darkorange', linewidth=1.5, label='Moving Avg (100)')

    ax2.set_title("Training Reward History")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)

    # ────────────────────────────────
    # 애니메이션 업데이트 함수
    # ────────────────────────────────
    def update(frame):
        drone_dot.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
        path_line.set_data(trajectory[:frame + 1, 0],
                           trajectory[:frame + 1, 1])
        return drone_dot, path_line

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(trajectory),
        interval=40,        # 40ms = 25fps
        blit=True,
        repeat=False
    )

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    evaluate_and_animate()
