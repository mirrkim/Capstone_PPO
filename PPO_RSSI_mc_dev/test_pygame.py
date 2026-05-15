import numpy as np
import torch
import os
import pygame
import scipy.ndimage
import gymnasium as gym
from gymnasium import spaces

from env import (DroneEnv, MAP_SIZE, 
                 BPSK_SIGNAL_RADIUS, QAM_SIGNAL_RADIUS,
                 TARGET_MODE)
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

# ────────────────────────────────────────────────────────
# 🎨 유틸리티: 예쁜 마커 및 맵 그리기 함수들
# ────────────────────────────────────────────────────────
def update_bg_surface(env):
    # 배경을 하얀색(255,255,255)으로 초기화
    screen_array = np.full((MAP_SIZE, MAP_SIZE, 3), 255, dtype=np.uint8)
    
    math_ys, math_xs = np.where(env.occupancy)
    pygame_xs = math_xs
    pygame_ys = MAP_SIZE - 1 - math_ys
    
    # 장애물 색상을 Matplotlib 감성의 부드러운 회색으로 칠하기
    screen_array[pygame_xs, pygame_ys] = [136, 136, 136] 
    return pygame.surfarray.make_surface(screen_array)

def draw_star(surface, color, center, radius):
    cx, cy = center
    points = []
    for i in range(10):
        angle = i * (np.pi / 5) - (np.pi / 2)
        r = radius if i % 2 == 0 else radius / 2.5
        points.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
    pygame.draw.polygon(surface, color, points)

def draw_x_marker(surface, color, center, size, width=3):
    cx, cy = center
    pygame.draw.line(surface, color, (cx - size, cy - size), (cx + size, cy + size), width)
    pygame.draw.line(surface, color, (cx - size, cy + size), (cx + size, cy - size), width)


# ────────────────────────────────────────────────────────
# 🚀 메인 실행 함수
# ────────────────────────────────────────────────────────
def evaluate_with_pygame():
    gym_env = GymDroneEnv()
    env = gym_env.env 
    agent = PPO(env.state_dim, env.action_dim)

    best_path = 'ppo_drone_best.pth'
    final_path = 'ppo_drone_final.pth'
    load_path = best_path if os.path.exists(best_path) else final_path

    try:
        agent.policy.load_state_dict(torch.load(load_path, map_location='cpu'))
        print(f"── 성공: '{load_path}' 가중치 로드 완료 ──")
    except FileNotFoundError:
        print(f"에러: 모델 파일이 없습니다.")
        return

    pygame.init()
    screen = pygame.display.set_mode((MAP_SIZE, MAP_SIZE))
    pygame.display.set_caption("Drone Interactive RL Demo (Matplotlib Style)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20, bold=True)

    # 🌟 [필수] 맵을 화면에 그리기 전에 무조건 reset()부터 실행!
    state, _ = gym_env.reset()
    device = next(agent.policy.parameters()).device
    trajectory = []

    # 1. 맵 배경 그리기 (하얀 배경 + 회색 장애물)
    bg_surface = update_bg_surface(env)

    # ────────────────────────────────────────────────────────
    # 🎨 [재수정] 신호 구역 그리기 (Matplotlib 감성의 부드러운 반투명 칠!)
    # ────────────────────────────────────────────────────────
    alpha_surface = pygame.Surface((MAP_SIZE, MAP_SIZE), pygame.SRCALPHA)
    
    # [1층] QAM 재머 (연한 파란색 칠 + X 마커)
    for qpos in env.qam_pos:
        qx, qy = int(qpos[0]), int(MAP_SIZE - qpos[1])
        # 🚀 속을 부드럽게 꽉 채운 반투명 파란색 (Alpha: 25)
        # 테두리 두께 옵션인 맨 끝의 '2'를 삭제했습니다.
        pygame.draw.circle(alpha_surface, (100, 100, 255, 25), (qx, qy), QAM_SIGNAL_RADIUS)
        # 예쁜 X 마커 (얘는 진하게)
        draw_x_marker(alpha_surface, (0, 0, 255, 255), (qx, qy), 8)

    # [2층] BPSK 타겟 (연한 빨간색 칠 + 별 마커) -> 파란색보다 위에 그려짐!
    tx, ty = int(env.bpsk_pos[0]), int(MAP_SIZE - env.bpsk_pos[1])
    # 🚀 속을 부드럽게 꽉 채운 반투명 빨간색 (Alpha: 35 - 조금 더 시인성 좋게)
    pygame.draw.circle(alpha_surface, (255, 100, 100, 35), (tx, ty), BPSK_SIGNAL_RADIUS)
    # 예쁜 빨간 별 마커 (얘는 진하게)
    draw_star(alpha_surface, (255, 0, 0, 255), (tx, ty), 12)

    running = True
    simulation_active = True
    final_msg = ""

    print("\n=== 라이브 시뮬레이션 시작! ===")
    print("👉 화면을 '마우스 좌클릭' 하면 실시간으로 장애물이 생성됩니다.")

    # ────────────────────────────────────────────────────────
    # 🔄 메인 게임 루프
    # ────────────────────────────────────────────────────────
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # 마우스 클릭 시 실시간 장애물 생성 및 SDF(거리 지도) 갱신
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = pygame.mouse.get_pos()
                map_x, map_y = mx, MAP_SIZE - my
                
                obs_radius = 30
                y_grid, x_grid = np.ogrid[0:MAP_SIZE, 0:MAP_SIZE]
                mask = (x_grid - map_x)**2 + (y_grid - map_y)**2 <= obs_radius**2
                
                env.occupancy[mask] = True
                empty_space = ~env.occupancy
                env.sdf_map = scipy.ndimage.distance_transform_edt(empty_space).astype(np.float32)
                
                # 새로 생긴 장애물을 반영하여 화면 다시 그리기
                bg_surface = update_bg_surface(env)
                print(f"🚨 라이브 장애물 투하: X={map_x}, Y={map_y} (경로 재탐색 중...)")

        # 드론 비행 연산
        if simulation_active:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) 
            with torch.no_grad():
                mu = agent.policy.actor(state_tensor)
            action = mu.cpu().numpy()[0] 

            next_state, reward, done, truncated, info = gym_env.step(action)
            state = next_state
            
            px, py = int(env.drone_pos[0]), int(MAP_SIZE - env.drone_pos[1])
            trajectory.append((px, py))

            if done or truncated:
                print(f"\n✅ 비행 종료: {info['msg']} (총 {env.steps} 스텝)")
                simulation_active = False
                final_msg = info['msg']

        # 화면 합성 및 출력
        screen.blit(bg_surface, (0, 0))    
        screen.blit(alpha_surface, (0, 0)) 

        # 시작 지점 그리기 (초록색 원)
        sx, sy = int(env.drone_start[0]), int(MAP_SIZE - env.drone_start[1])
        pygame.draw.circle(screen, (0, 128, 0), (sx, sy), 8)

        # 드론 궤적(꼬리) 그리기 (분홍색)
        if len(trajectory) > 1:
            pygame.draw.lines(screen, (255, 0, 255), False, trajectory, 2)

        # 라이다(녹색 선) 그리기
        lidar_vals = env._cast_lidar_rays(env.drone_pos)
        _angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        for i, dist_norm in enumerate(lidar_vals):
            dist = dist_norm * 200.0 
            end_x = int(env.drone_pos[0] + dist * np.cos(_angles[i]))
            end_y = int(MAP_SIZE - (env.drone_pos[1] + dist * np.sin(_angles[i]))) 
            pygame.draw.line(screen, (0, 200, 0), (px, py), (end_x, end_y), 1)

        # 드론 본체 그리기
        pygame.draw.circle(screen, (255, 0, 255), (px, py), 8)
        pygame.draw.circle(screen, (255, 255, 255), (px, py), 4)

        # HUD 정보 출력창
        curr_rssi = env.get_bpsk_rssi(env.drone_pos)
        hud_text = font.render(f"Steps: {env.steps}/512 | RSSI: {curr_rssi:.2f}", True, (0, 0, 0))
        screen.blit(hud_text, (10, 10))

        # 종료 메시지 출력
        if not simulation_active:
            end_text = font.render(f"FINISHED: {final_msg}", True, (255, 0, 0))
            screen.blit(end_text, (10, 40))

        pygame.display.flip()
        clock.tick(30) # 30FPS 로 속도 조절

    pygame.quit()

if __name__ == '__main__':
    evaluate_with_pygame()