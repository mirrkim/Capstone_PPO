"""
test_editor.py  —  pygame 장애물 에디터 + 드론 시뮬레이터

사용법:
  1. 도구를 선택해 장애물(원/직사각형)을 그립니다.
  2. Start / BPSK 위치를 원하는 곳에 클릭해 설정합니다.
  3. [▶ Start Simulation] 버튼을 눌러 비행을 시작합니다.
  4. [◀ Back to Editor] 로 돌아와 맵을 수정한 뒤 재시작할 수 있습니다.
"""
import sys
import os
import heapq
import itertools
import pygame
import numpy as np
import torch
import scipy.ndimage

# Localization 버튼 클릭 시 호출할 실제 삼변측량 모듈
from localization import run_localization


from env import (DroneEnv, MAP_SIZE,
                 BPSK_SIGNAL_RADIUS, QAM_SIGNAL_RADIUS,
                 TARGET_RSSI_THRESHOLD,
                 NUM_LIDAR_RAYS, LIDAR_MAX_RANGE, _QAM_POS)
from ppo import PPO

# ─────────────────────────────────────────────────────────────────────
#  색상 팔레트
# ─────────────────────────────────────────────────────────────────────
MAP_BG      = (248, 248, 244)
OBS_COL     = (55,  58,  70)
PANEL_BG    = (36,  40,  52)
DARK_BG     = (28,  30,  40)
GRID_COL    = (180, 180, 175)

DRONE_COL   = (220, 50,  180)
FOLLOWER_COL_1 = (255, 135, 210)
FOLLOWER_COL_2 = (180, 90, 255)
FORMATION_LINE_COL = (255, 210, 245)
TRAJ_COL    = (220, 50,  180)
BPSK_COL    = (210, 60,  60)
QAM_COL     = (60,  110, 220)
LOC_BASE_COL = (230, 45, 45)      # Baseline: red
LOC_KAL_COL  = (0, 120, 255)       # Kalman: blue
LOC_ANN_COL  = (40, 190, 80)       # Kalman+ANN: green
LOC_CIRCLE_BASE = (230, 45, 45)
LOC_CIRCLE_KAL  = (0, 120, 255)
LOC_CIRCLE_ANN  = (40, 190, 80)
START_COL   = (50,  200, 80)
PREV_COL    = (160, 160, 215)

BTN_ACTIVE  = (70,  130, 200)
BTN_NORMAL  = (52,  58,  76)
TITLE_COL   = (240, 245, 255)
TEXT_COL    = (205, 210, 222)
DIM_COL     = (140, 148, 162)
OK_COL      = (80,  200, 100)
ERR_COL     = (210, 75,  75)
WARN_COL    = (220, 180, 60)
WHITE       = (255, 255, 255)

# ─────────────────────────────────────────────────────────────────────
#  레이아웃
# ─────────────────────────────────────────────────────────────────────
MAP_PX  = 700       # 맵 표시 크기 (env MAP_SIZE 와 1:1)
PANEL_W = 320       # 오른쪽 패널 너비
WIN_W   = MAP_PX + PANEL_W
WIN_H   = MAP_PX

# ─────────────────────────────────────────────────────────────────────
#  도구 ID
# ─────────────────────────────────────────────────────────────────────
T_CIRCLE = 'circle'
T_RECT   = 'rect'
T_ERASER = 'eraser'
T_START  = 'start'
T_BPSK   = 'bpsk'

# ─────────────────────────────────────────────────────────────────────
#  Mission settings
# ─────────────────────────────────────────────────────────────────────
MISSION_RSSI_THRESHOLD = TARGET_RSSI_THRESHOLD  # env.py의 도착 기준을 그대로 사용
FOLLOWER_GAP = 90.0             # 리더-F1-F2 사이 기본 간격(px)
FOLLOWER_CATCHUP_SPEED = 4.0    # 리더 도착 후 팔로워가 경로를 따라 따라붙는 속도(px/frame)

# Localization 버튼을 누른 뒤 삼각대형으로 흩어지는 단계 설정
FORMATION_RADII = (95.0, 120.0, 145.0)       # 드론들이 모인 중심 기준 정삼각형 후보 반지름(px)
FORMATION_ANGLE_STEP_DEG = 15              # 정삼각형 후보 회전 간격(deg)
FORMATION_SAFE_MARGIN = 12.0               # 꼭짓점이 장애물에서 최소 이 정도는 떨어져야 함(px)
FORMATION_GRID_STEP = 10                   # A* 격자 해상도(px/cell)
FORMATION_MOVE_SPEED = 5.0                 # 삼각대형 이동 속도(px/frame)

# ─────────────────────────────────────────────────────────────────────
#  좌표 변환 헬퍼
#  env:    (0,0) = 맵 왼쪽 아래,  y 위쪽이 큼
#  pygame: (0,0) = 화면 왼쪽 위,  y 아래쪽이 큼
# ─────────────────────────────────────────────────────────────────────
def e2s(ex, ey):
    """env 좌표 → 화면(pygame) 좌표"""
    return int(ex), MAP_PX - 1 - int(ey)

def s2e(sx, sy):
    """화면 좌표 → env 좌표"""
    return float(sx), float(MAP_PX - 1 - sy)


# ─────────────────────────────────────────────────────────────────────
#  간단한 버튼 클래스
# ─────────────────────────────────────────────────────────────────────
class Button:
    def __init__(self, rect, label, active=False):
        self.rect   = pygame.Rect(rect)
        self.label  = label
        self.active = active

    def draw(self, surf, font):
        col = BTN_ACTIVE if self.active else BTN_NORMAL
        pygame.draw.rect(surf, col,             self.rect, border_radius=6)
        pygame.draw.rect(surf, (90, 98, 120),   self.rect, 1, border_radius=6)
        txt = font.render(self.label, True, WHITE)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def hit(self, ev):
        return (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and self.rect.collidepoint(ev.pos))


# ─────────────────────────────────────────────────────────────────────
#  메인 앱
# ─────────────────────────────────────────────────────────────────────
class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption('Drone Editor & Simulator')
        self.clock  = pygame.time.Clock()

        self.fn_s  = pygame.font.SysFont('consolas', 13)
        self.fn_m  = pygame.font.SysFont('consolas', 15)
        self.fn_l  = pygame.font.SysFont('consolas', 19, bold=True)
        self.fn_xl = pygame.font.SysFont('consolas', 23, bold=True)
        # 결과 해석 문구는 한글 표시를 위해 맑은 고딕을 우선 사용
        self.fn_kr_s = pygame.font.SysFont('malgungothic', 13)
        self.fn_kr_m = pygame.font.SysFont('malgungothic', 14)

        self._init_editor()

    # =================================================================
    #  에디터 초기화
    # =================================================================
    def _init_editor(self):
        self.mode = 'editor'

        # 점유 격자 (에디터에서 직접 그리는 맵)
        self.occ        = np.zeros((MAP_PX, MAP_PX), dtype=bool)
        self.start_pos  = np.array([30.0,  30.0])
        self.bpsk_pos   = np.array([620.0, 620.0])

        self.tool        = T_CIRCLE
        self.brush_r     = 50      # 원 반지름 / 직사각형 절반 크기
        self.eraser_r    = 40

        self.dragging    = False
        self.drag_start  = None    # 화면 좌표

        self._map_surf   = None    # 격자 렌더 캐시
        self._map_dirty  = True
        self._voff = np.array([0, 0], dtype=float)  # 뷰포트 오프셋

        # ── 도구 버튼 ────────────────────────────────────────────────
        px = MAP_PX + 14
        self.tool_btns = [
            Button((px,       22, 140, 34), '○  Circle',   self.tool == T_CIRCLE),
            Button((px + 148, 22, 140, 34), '□  Rect',     self.tool == T_RECT),
            Button((px,       62, 140, 34), '✕  Eraser',   self.tool == T_ERASER),
            Button((px + 148, 62, 140, 34), '◎  Start',    self.tool == T_START),
            Button((px,      102, 140, 34), '★  BPSK',     self.tool == T_BPSK),
        ]
        self._tool_ids = [T_CIRCLE, T_RECT, T_ERASER, T_START, T_BPSK]

        self.btn_clear = Button((px, 148, 290, 34), '🗑  Clear All')
        self.btn_run   = Button((px, WIN_H - 58, 290, 46), '▶  Start Simulation')

    # =================================================================
    #  점유 격자 → 렌더 서피스 (캐시)
    # =================================================================
    def _get_map_surf(self, occ=None):
        """occ 파라미터가 주어지면 캐시 없이 즉시 생성"""
        src = occ if occ is not None else self.occ
        dirty = (occ is not None) or self._map_dirty or self._map_surf is None
        if not dirty:
            return self._map_surf

        surf = pygame.Surface((MAP_PX, MAP_PX))
        surf.fill(MAP_BG)
        # occupancy[y, x]: y=0 이 맵 아래쪽 → 화면에서는 flipud 필요
        flipped = np.flipud(src)                          # (H, W)
        pixels  = np.where(
            flipped.T[:, :, None],                        # (W, H, 1) broadcast
            np.array(OBS_COL, dtype=np.uint8),
            np.array(MAP_BG,  dtype=np.uint8)
        )                                                  # (W, H, 3)
        pygame.surfarray.blit_array(surf, pixels)

        if occ is None:
            self._map_surf  = surf
            self._map_dirty = False
        return surf

    # =================================================================
    #  격자에 도형 그리기 / 지우기
    # =================================================================
    def _paint_circle(self, cx, cy, r, erase=False):
        yg, xg = np.ogrid[0:MAP_PX, 0:MAP_PX]
        mask = (xg - cx) ** 2 + (yg - cy) ** 2 <= r ** 2
        self.occ[mask]  = not erase
        self._map_dirty = True

    def _paint_rect(self, x0, y0, x1, y1, erase=False):
        xi0 = int(np.clip(min(x0, x1), 0, MAP_PX - 1))
        xi1 = int(np.clip(max(x0, x1), 0, MAP_PX))
        yi0 = int(np.clip(min(y0, y1), 0, MAP_PX - 1))
        yi1 = int(np.clip(max(y0, y1), 0, MAP_PX))
        self.occ[yi0:yi1, xi0:xi1] = not erase
        self._map_dirty = True

    # =================================================================
    #  에디터 이벤트 처리
    # =================================================================
    def _editor_events(self, events):
        for ev in events:
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            # 도구 선택
            for i, btn in enumerate(self.tool_btns):
                if btn.hit(ev):
                    self.tool = self._tool_ids[i]
                    for b in self.tool_btns:
                        b.active = False
                    btn.active = True

            if self.btn_clear.hit(ev):
                self.occ[:] = False
                self._map_dirty = True

            if self.btn_run.hit(ev):
                self._launch()

            # 마우스 버튼 누름 (맵 영역)
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                sx, sy = ev.pos
                if sx < MAP_PX:
                    self.dragging   = True
                    self.drag_start = (sx, sy)
                    ex, ey = s2e(sx, sy)
                    if self.tool == T_START:
                        self.start_pos = np.array([ex, ey])
                    elif self.tool == T_BPSK:
                        self.bpsk_pos = np.array([ex, ey])
                    elif self.tool == T_ERASER:
                        self._paint_circle(ex, ey, self.eraser_r, erase=True)

            # 마우스 버튼 뗌 → 도형 확정
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                if self.dragging and self.drag_start:
                    sx0, sy0 = self.drag_start
                    sx1, sy1 = ev.pos
                    if sx1 < MAP_PX and sx0 < MAP_PX:
                        ex0, ey0 = s2e(sx0, sy0)
                        ex1, ey1 = s2e(sx1, sy1)
                        if self.tool == T_CIRCLE:
                            r = np.hypot(ex1 - ex0, ey1 - ey0)
                            if r > 5:
                                self._paint_circle(ex0, ey0, r)
                        elif self.tool == T_RECT:
                            if abs(ex1 - ex0) > 4 and abs(ey1 - ey0) > 4:
                                self._paint_rect(ex0, ey0, ex1, ey1)
                self.dragging   = False
                self.drag_start = None

            # 드래그 중 지우개
            if ev.type == pygame.MOUSEMOTION and self.dragging:
                sx, sy = ev.pos
                if sx < MAP_PX and self.tool == T_ERASER:
                    ex, ey = s2e(sx, sy)
                    self._paint_circle(ex, ey, self.eraser_r, erase=True)

            # 마우스 휠 → 브러시 크기
            if ev.type == pygame.MOUSEWHEEL:
                self.brush_r  = int(np.clip(self.brush_r  + ev.y * 4, 8, 180))
                self.eraser_r = int(np.clip(self.eraser_r + ev.y * 4, 8, 180))

    # =================================================================
    #  에디터 렌더링
    # =================================================================
    def _render_editor(self):
        self.screen.fill(DARK_BG)

        # 맵
        self.screen.blit(self._get_map_surf(), (0, 0))

        # 격자 구분선 (3×3)
        for i in range(1, 3):
            v = MAP_PX * i // 3
            pygame.draw.line(self.screen, GRID_COL, (v, 0), (v, MAP_PX), 1)
            pygame.draw.line(self.screen, GRID_COL, (0, v), (MAP_PX, v), 1)

        # BPSK 타겟
        bsx, bsy = e2s(*self.bpsk_pos)
        pygame.draw.circle(self.screen, BPSK_COL, (bsx, bsy), BPSK_SIGNAL_RADIUS, 1)
        pygame.draw.circle(self.screen, BPSK_COL, (bsx, bsy), 9)
        self.screen.blit(self.fn_s.render('BPSK', True, BPSK_COL), (bsx + 11, bsy - 8))

        # QAM 방해꾼
        for qp in _QAM_POS:
            qsx, qsy = self._vs(qp[0], qp[1])
            pygame.draw.circle(self.screen, QAM_COL, (qsx, qsy), QAM_SIGNAL_RADIUS, 1)
            pygame.draw.circle(self.screen, QAM_COL, (qsx, qsy), 7)

        # 드론 시작 위치
        ssx, ssy = e2s(*self.start_pos)
        pygame.draw.circle(self.screen, START_COL, (ssx, ssy), 10)
        pygame.draw.circle(self.screen, WHITE,     (ssx, ssy), 10, 2)
        self.screen.blit(self.fn_s.render('Start', True, START_COL), (ssx + 12, ssy - 8))

        # 드래그 프리뷰
        mx, my = pygame.mouse.get_pos()
        if mx < MAP_PX:
            if self.tool == T_ERASER:
                pygame.draw.circle(self.screen, (200, 100, 100), (mx, my), self.eraser_r, 2)
            elif self.dragging and self.drag_start:
                sx0, sy0 = self.drag_start
                if self.tool == T_CIRCLE:
                    r = int(np.hypot(mx - sx0, my - sy0))
                    pygame.draw.circle(self.screen, PREV_COL, (sx0, sy0), r, 2)
                elif self.tool == T_RECT:
                    rx, ry = min(sx0, mx), min(sy0, my)
                    pygame.draw.rect(self.screen, PREV_COL,
                                     (rx, ry, abs(mx-sx0), abs(my-sy0)), 2)
            elif self.tool in (T_CIRCLE, T_RECT):
                pygame.draw.circle(self.screen, (170, 170, 210), (mx, my), self.brush_r, 1)

        # ── 패널 ─────────────────────────────────────────────────────
        px = MAP_PX
        pygame.draw.rect(self.screen, PANEL_BG, (px, 0, PANEL_W, WIN_H))
        pygame.draw.line(self.screen, (65, 70, 90), (px, 0), (px, WIN_H), 2)

        # 제목
        self.screen.blit(self.fn_xl.render('Obstacle Editor', True, TITLE_COL), (px + 12, 6))

        # 버튼
        for btn in self.tool_btns:
            btn.draw(self.screen, self.fn_m)
        self.btn_clear.draw(self.screen, self.fn_m)
        self.btn_run.draw(self.screen, self.fn_l)

        # 현재 도구 표시
        tool_names = {T_CIRCLE:'Circle', T_RECT:'Rect',
                      T_ERASER:'Eraser', T_START:'Start Pos', T_BPSK:'BPSK Target'}
        self.screen.blit(
            self.fn_m.render(f'Active: {tool_names[self.tool]}', True, BTN_ACTIVE),
            (px + 14, 142))

        # 조작 안내
        iy = 195
        for line, col in [
            ('  Circle / Rect : drag to size',    DIM_COL),
            ('  Eraser        : click & drag',     DIM_COL),
            ('  Start / BPSK  : click to place',   DIM_COL),
            ('  Scroll wheel  : change brush size',DIM_COL),
        ]:
            self.screen.blit(self.fn_s.render(line, True, col), (px + 12, iy))
            iy += 18

        # 현재 설정 요약
        iy += 8
        pygame.draw.line(self.screen, (58, 63, 82), (px+8, iy), (px+PANEL_W-8, iy))
        iy += 10
        self.screen.blit(self.fn_m.render('Settings', True, TITLE_COL), (px + 14, iy)); iy += 24
        self.screen.blit(
            self.fn_s.render(f'  Brush size : {self.brush_r} px', True, TEXT_COL),
            (px + 14, iy)); iy += 18
        self.screen.blit(
            self.fn_s.render(f'  Start pos  : ({self.start_pos[0]:.0f}, {self.start_pos[1]:.0f})',
                             True, START_COL), (px + 14, iy)); iy += 18
        self.screen.blit(
            self.fn_s.render(f'  BPSK pos   : ({self.bpsk_pos[0]:.0f}, {self.bpsk_pos[1]:.0f})',
                             True, BPSK_COL), (px + 14, iy)); iy += 18
        obs_pct = self.occ.mean() * 100
        self.screen.blit(
            self.fn_s.render(f'  Obstacles  : {obs_pct:.1f}% of map', True, TEXT_COL),
            (px + 14, iy))

        pygame.display.flip()

    # =================================================================
    #  뷰포트 좌표 변환 (_voff 적용)
    # =================================================================
    def _vs(self, ex, ey):
        """env 좌표 → 화면 좌표 (뷰포트 오프셋 포함)"""
        sx, sy = e2s(ex, ey)
        return int(sx + self._voff[0]), int(sy + self._voff[1])

    # =================================================================
    #  시뮬레이션 시작
    # =================================================================
    def _launch(self):
        # ── 모델 로드 ────────────────────────────────────────────────
        best_path  = 'ppo_drone_best.pth'
        final_path = 'ppo_drone_final.pth'
        if   os.path.exists(best_path):  load_path = best_path
        elif os.path.exists(final_path): load_path = final_path
        else:                            load_path = None

        # PPO 탐색 종료 기준은 env.py의 TARGET_RSSI_THRESHOLD 값을 사용합니다.
        self.env = DroneEnv()

        # ── 커스텀 맵 주입 ─────────────────────────────────────────
        # _generate_world() 를 람다로 교체해 에디터 맵을 직접 사용합니다.
        # (env.drone_start 는 공개 속성이라 외부에서 바로 대입 가능)
        snap_occ   = self.occ.copy()
        snap_bpsk  = self.bpsk_pos.copy()

        # 원본 env._generate_world 와 동일하게 외곽 벽도 True 로 처리
        snap_occ[0:2,  :] = True
        snap_occ[-2:,  :] = True
        snap_occ[:,  0:2] = True
        snap_occ[:, -2:]  = True

        def _custom_world():
            self.env.occupancy = snap_occ.copy()
            self.env.sdf_map   = scipy.ndimage.distance_transform_edt(
                ~snap_occ).astype(np.float32)
            self.env.bpsk_pos  = snap_bpsk.copy()

        self.env._generate_world = _custom_world
        self.env.drone_start     = self.start_pos.copy()

        # ── PPO 에이전트 ──────────────────────────────────────────────
        self.agent = PPO(self.env.state_dim, self.env.action_dim)
        if load_path:
            self.agent.policy.load_state_dict(
                torch.load(load_path, map_location='cpu'))
            self._model_label = load_path
        else:
            self._model_label = 'random policy (no model found)'
        self.agent.policy.eval()
        self.device = next(self.agent.policy.parameters()).device

        # ── 상태 초기화 ───────────────────────────────────────────────
        init_state          = self.env.reset()
        self.sim_state      = init_state
        self.trajectory     = [self.env.drone_pos.copy()]
        self.total_reward   = 0.0
        self.last_reward    = 0.0
        self.step_count     = 0
        self.sim_done       = False
        self.sim_result     = ''
        self.leader_arrived = False
        self.follower_catchup_progress = 0.0
        self.follower_catchup_steps = 0
        self.final_drone_positions = None
        self.localization_result = None
        self.formation_drone_positions = None
        self.formation_targets = None
        self.formation_paths = None
        self.formation_path_indices = None
        self.formation_center = None
        self.formation_radius = None
        self._lidar         = np.ones(NUM_LIDAR_RAYS, dtype=np.float32)

        # 맵 배경 서피스 1회 생성 (시뮬레이션 중 불변)
        self._sim_bg = self._get_map_surf(occ=self.env.occupancy)

        self.mode = 'running'

        # 뒤로 버튼
        self.btn_back         = Button((MAP_PX + 12, WIN_H - 58, 292, 46), '◀  Back to Editor')
        self.btn_localization = Button((MAP_PX + 12, WIN_H - 112, 292, 46), '🔍  Localization')

    # =================================================================
    #  시뮬레이션 스텝 (1 프레임에 1 스텝)
    # =================================================================
    def _sim_step(self):
        if self.sim_done:
            return
        st = torch.FloatTensor(self.sim_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu = self.agent.policy.actor(st)
        action = mu.cpu().numpy()[0]

        next_state, reward, done, msg = self.env.step(action)
        self.sim_state   = next_state
        self.last_reward = reward
        self.total_reward += reward
        self.step_count  += 1
        self.trajectory.append(self.env.drone_pos.copy())
        # 상태 벡터 끝 NUM_LIDAR_RAYS 개 = 라이다 값
        self._lidar = next_state[-NUM_LIDAR_RAYS:]

        if done:
            if msg == "Target Found":
                # PPO 입장에서는 성공 종료지만, 전체 미션은 아직 끝내지 않음.
                # 리더는 현재 위치에서 정지하고 F1/F2가 같은 안전 경로를 따라 들어오게 한다.
                self.leader_arrived = True
                self.sim_done = False
                self.sim_result = "Leader Arrived"
                self.mode = 'followers_catchup'
            else:
                self.sim_done   = True
                self.sim_result = msg
                self.mode       = 'done'

    # =================================================================
    #  시뮬레이션 이벤트
    # =================================================================
    def _sim_events(self, events):
        for ev in events:
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if self.btn_back.hit(ev):
                self._init_editor()
            if (self.mode == 'ready_localization'
                    and self.btn_localization.hit(ev)):
                self._start_formation_move()


    # =================================================================
    #  리더 도착 후 팔로워 추종 단계
    # =================================================================
    def _followers_catchup_step(self):
        """
        리더가 Target Found에 도달한 뒤에는 PPO를 더 돌리지 않는다.
        대신 F1/F2가 리더가 이미 지나간 trajectory 위를 따라오도록
        delay distance를 프레임마다 줄인다.
        """
        if self.mode != 'followers_catchup':
            return

        self.follower_catchup_progress += FOLLOWER_CATCHUP_SPEED
        self.follower_catchup_steps += 1

        formation_positions = self._get_formation_positions()
        last_follower = formation_positions[2]
        last_rssi = self.env.get_bpsk_rssi(last_follower)
        last_dist = float(np.linalg.norm(last_follower - self.env.bpsk_pos))

        # 마지막 드론(F2)이 같은 목표 판정 기준(0.6)에 들어오면 전체 비행 완료
        if last_rssi >= MISSION_RSSI_THRESHOLD:
            self.sim_done = True
            self.sim_result = 'All Drones Arrived'
            self.mode = 'ready_localization'
            self.final_drone_positions = np.array(formation_positions, dtype=float)
            print('=' * 55)
            print('[Mission] All drones arrived')
            print(f'  threshold RSSI : {MISSION_RSSI_THRESHOLD:.2f}')
            print(f'  F2 RSSI        : {last_rssi:.3f}')
            print(f'  F2 distance    : {last_dist:.1f}px')
            print(f'  drone positions: {self.final_drone_positions}')
            print('=' * 55)

    # =================================================================
    #  긴 간격 시간차 추종 위치 계산: 리더 1대 + 팔로워 2대
    # =================================================================
    def _get_trail_position(self, delay_distance):
        """
        리더가 이미 지나간 trajectory 위에서 delay_distance만큼 뒤쪽 위치를 찾습니다.
        팔로워가 리더의 안전 경로를 그대로 따라오게 하므로,
        간격을 길게 둔 시간차 추종이라 단순 삼각 편대보다 장애물 충돌 가능성이 낮습니다.
        """
        if not hasattr(self, 'trajectory') or len(self.trajectory) == 0:
            return self.env.drone_pos.copy()

        traj = self.trajectory
        # 리더 도착 전에는 고정 간격, 리더 도착 후에는 progress만큼 delay를 줄여 팔로워가 따라붙게 함
        if getattr(self, 'leader_arrived', False):
            remaining = max(0.0, float(delay_distance) - float(self.follower_catchup_progress))
        else:
            remaining = float(delay_distance)

        # 현재 위치에서 과거 궤적 방향으로 거리를 누적하며 뒤로 이동
        for i in range(len(traj) - 1, 0, -1):
            p_now = traj[i]
            p_prev = traj[i - 1]
            seg = p_now - p_prev
            seg_len = float(np.linalg.norm(seg))

            if seg_len < 1e-6:
                continue

            if remaining <= seg_len:
                # p_now에서 p_prev 방향으로 remaining만큼 이동한 점
                ratio = remaining / seg_len
                return p_now + (p_prev - p_now) * ratio

            remaining -= seg_len

        # 아직 리더가 충분히 멀리 가지 않았다면 시작점에 대기
        return traj[0].copy()

    def _get_formation_positions(self):
        """
        화면 표시용 3대 위치.
        L은 PPO가 실제 제어하는 드론, F1/F2는 L의 경로를 시간차로 따라옵니다.
        Localization 버튼 이후에는 A*로 이동 중인 실제 삼각대형 위치를 반환합니다.
        """
        if getattr(self, 'formation_drone_positions', None) is not None and self.mode in ('formation_move', 'done'):
            return [p.copy() for p in self.formation_drone_positions]

        leader = self.env.drone_pos.copy()

        follower_gap = FOLLOWER_GAP  # 드론 간 거리(px)
        follower_1 = self._get_trail_position(follower_gap)
        follower_2 = self._get_trail_position(follower_gap * 2.0)

        positions = [leader, follower_1, follower_2]
        positions = [np.clip(pos, 3.0, self.env.map_size - 3.0) for pos in positions]
        return positions


    # =================================================================
    #  Localization 전 삼각대형 재배치 단계
    # =================================================================
    def _is_safe_point(self, p, margin=FORMATION_SAFE_MARGIN):
        """맵 안이고 장애물과 margin 이상 떨어진 점인지 확인."""
        x = int(np.clip(round(float(p[0])), 0, self.env.map_size - 1))
        y = int(np.clip(round(float(p[1])), 0, self.env.map_size - 1))
        if x <= 2 or y <= 2 or x >= self.env.map_size - 3 or y >= self.env.map_size - 3:
            return False
        if self.env.occupancy[y, x]:
            return False
        return float(self.env.sdf_map[y, x]) >= float(margin)

    def _nearest_safe_point(self, p, max_radius=80):
        """후보 꼭짓점이 장애물에 걸리면 주변의 가장 가까운 안전 지점으로 보정."""
        p = np.asarray(p, dtype=float)
        if self._is_safe_point(p):
            return np.clip(p, 3.0, self.env.map_size - 3.0)

        best = None
        best_score = float('inf')
        for r in range(5, max_radius + 1, 5):
            for deg in range(0, 360, 15):
                a = np.deg2rad(deg)
                q = p + r * np.array([np.cos(a), np.sin(a)])
                q = np.clip(q, 3.0, self.env.map_size - 3.0)
                if self._is_safe_point(q):
                    x = int(round(q[0])); y = int(round(q[1]))
                    # 원래 후보에서 가까우면서 장애물에서 멀수록 좋게 평가
                    score = np.linalg.norm(q - p) - 0.25 * float(self.env.sdf_map[y, x])
                    if score < best_score:
                        best_score = score
                        best = q.copy()
            if best is not None:
                return best
        return np.clip(p, 3.0, self.env.map_size - 3.0)

    def _make_blocked_grid(self, step=FORMATION_GRID_STEP, margin=FORMATION_SAFE_MARGIN):
        """occupancy를 A*용 coarse grid로 변환. 장애물 주변 margin까지 막힌 칸으로 확장."""
        iterations = max(1, int(np.ceil(float(margin) / float(step))))
        small_occ = self.env.occupancy[::step, ::step]
        blocked = scipy.ndimage.binary_dilation(small_occ, iterations=iterations)
        return blocked.astype(bool)

    def _cell_from_pos(self, p, blocked, step=FORMATION_GRID_STEP):
        h, w = blocked.shape
        cx = int(np.clip(round(float(p[0]) / step), 0, w - 1))
        cy = int(np.clip(round(float(p[1]) / step), 0, h - 1))
        if not blocked[cy, cx]:
            return (cx, cy)
        # 시작/목표가 확장 장애물에 살짝 들어가면 가장 가까운 빈 칸으로 보정
        for rad in range(1, max(h, w)):
            best = None
            best_d = float('inf')
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and not blocked[ny, nx]:
                        d = dx * dx + dy * dy
                        if d < best_d:
                            best_d = d
                            best = (nx, ny)
            if best is not None:
                return best
        return (cx, cy)

    def _pos_from_cell(self, cell, step=FORMATION_GRID_STEP):
        x, y = cell
        return np.array([x * step, y * step], dtype=float)

    def _astar_path(self, start, goal, blocked, step=FORMATION_GRID_STEP):
        """coarse grid A* 경로. 반환값은 env px 좌표 배열 또는 None."""
        start_c = self._cell_from_pos(start, blocked, step)
        goal_c = self._cell_from_pos(goal, blocked, step)
        h, w = blocked.shape

        def heuristic(a, b):
            return float(np.hypot(a[0] - b[0], a[1] - b[1]))

        moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414),
        ]
        open_heap = [(heuristic(start_c, goal_c), 0.0, start_c)]
        came = {}
        gscore = {start_c: 0.0}
        closed = set()

        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == goal_c:
                cells = [cur]
                while cur in came:
                    cur = came[cur]
                    cells.append(cur)
                cells.reverse()
                pts = [np.asarray(start, dtype=float)]
                pts.extend([self._pos_from_cell(c, step) for c in cells[1:-1]])
                pts.append(np.asarray(goal, dtype=float))
                return self._smooth_path(np.array(pts, dtype=float))

            closed.add(cur)
            for dx, dy, cost in moves:
                nx, ny = cur[0] + dx, cur[1] + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if blocked[ny, nx]:
                    continue
                nxt = (nx, ny)
                ng = g + cost
                if ng < gscore.get(nxt, float('inf')):
                    gscore[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(open_heap, (ng + heuristic(nxt, goal_c), ng, nxt))
        return None

    def _segment_is_safe(self, a, b, margin=FORMATION_SAFE_MARGIN):
        """두 점 사이 직선이 장애물 margin 안으로 들어가지 않는지 확인."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        dist = float(np.linalg.norm(b - a))
        steps = max(2, int(dist / 4.0))
        ts = np.linspace(0.0, 1.0, steps)
        pts = a + ts[:, None] * (b - a)
        xs = np.clip(np.rint(pts[:, 0]).astype(int), 0, self.env.map_size - 1)
        ys = np.clip(np.rint(pts[:, 1]).astype(int), 0, self.env.map_size - 1)
        if np.any(self.env.occupancy[ys, xs]):
            return False
        return bool(np.min(self.env.sdf_map[ys, xs]) >= margin)

    def _smooth_path(self, pts):
        """A* 경로에서 직선으로 건너뛸 수 있는 중간점을 제거."""
        if len(pts) <= 2:
            return pts
        smoothed = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1:
                if self._segment_is_safe(pts[i], pts[j]):
                    break
                j -= 1
            smoothed.append(pts[j])
            i = j
        return np.array(smoothed, dtype=float)

    def _path_length(self, path):
        if path is None or len(path) < 2:
            return float('inf')
        return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

    def _triangle_points(self, center, radius, angle_offset):
        angles = np.array([angle_offset, angle_offset + 2*np.pi/3, angle_offset + 4*np.pi/3])
        pts = np.array([center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles], dtype=float)
        return np.array([self._nearest_safe_point(p) for p in pts], dtype=float)

    def _find_best_triangle_formation(self, drone_positions):
        """
        목표 좌표를 직접 쓰지 않고, 세 드론이 모인 위치의 평균점을 중심으로
        여러 정삼각형 후보를 만든 뒤 A* 경로 비용이 가장 작은 후보를 선택.
        """
        drone_positions = np.asarray(drone_positions, dtype=float)
        center = np.mean(drone_positions, axis=0)
        center = self._nearest_safe_point(center)
        blocked = self._make_blocked_grid()

        best = None
        best_score = float('inf')

        for radius in FORMATION_RADII:
            for deg in range(0, 120, FORMATION_ANGLE_STEP_DEG):
                pts = self._triangle_points(center, radius, np.deg2rad(deg))
                if not all(self._is_safe_point(p) for p in pts):
                    continue

                # 삼각형 꼭짓점이 너무 찌그러졌으면 제외
                side_lengths = [np.linalg.norm(pts[(i + 1) % 3] - pts[i]) for i in range(3)]
                if min(side_lengths) < radius * 1.0:
                    continue

                # 세 드론과 세 꼭짓점의 모든 배정 조합 중 A* 총 길이가 최소인 것 선택
                for perm in itertools.permutations(range(3)):
                    targets = pts[list(perm)]
                    paths = []
                    total_len = 0.0
                    feasible = True
                    for i in range(3):
                        path = self._astar_path(drone_positions[i], targets[i], blocked)
                        if path is None:
                            feasible = False
                            break
                        paths.append(path)
                        total_len += self._path_length(path)
                    if not feasible:
                        continue

                    safety = np.mean([self.env.sdf_map[int(round(p[1])), int(round(p[0]))] for p in targets])
                    score = total_len - 1.5 * safety
                    if score < best_score:
                        best_score = score
                        best = {
                            'center': center,
                            'radius': radius,
                            'targets': targets,
                            'paths': paths,
                            'score': score,
                            'path_length': total_len,
                        }

        if best is None:
            raise RuntimeError('안전한 삼각대형 A* 경로를 찾지 못했습니다. 장애물을 줄이거나 시작 위치를 조정하세요.')
        return best

    def _start_formation_move(self):
        """Localization 버튼 클릭 → 바로 삼변측량하지 않고 A* 삼각대형 배치부터 시작."""
        if self.final_drone_positions is None:
            self.final_drone_positions = np.array(self._get_formation_positions(), dtype=float)

        try:
            plan = self._find_best_triangle_formation(self.final_drone_positions)
        except Exception as e:
            print('[Formation Error]', repr(e))
            self.sim_result = f'Formation Error: {e}'
            return

        self.formation_drone_positions = self.final_drone_positions.copy()
        self.formation_targets = plan['targets']
        self.formation_paths = plan['paths']
        self.formation_path_indices = [0, 0, 0]
        self.formation_center = plan['center']
        self.formation_radius = plan['radius']
        self.sim_done = False
        self.sim_result = 'Formation Moving'
        self.mode = 'formation_move'

        # ── 뷰포트 오프셋: 목표를 화면 중앙으로 ──────────────────────
        tgt_sx, tgt_sy = e2s(*self.env.bpsk_pos)
        self._voff = np.array([MAP_PX // 2 - tgt_sx,
                               MAP_PX // 2 - tgt_sy], dtype=float)

        print('=' * 60)
        print('[Formation] A* 삼각대형 이동 시작')
        print(f'  center        : {np.round(self.formation_center, 2)}')
        print(f'  radius        : {self.formation_radius:.1f}px')
        print(f'  path length   : {plan["path_length"]:.1f}px')
        print(f'  targets       :\n{np.round(self.formation_targets, 2)}')
        print('=' * 60)

    def _move_along_path(self, pos, path, idx, speed=FORMATION_MOVE_SPEED):
        """현재 pos를 path[idx] 이후 목표점으로 speed만큼 이동."""
        pos = np.asarray(pos, dtype=float).copy()
        if path is None or len(path) == 0:
            return pos, idx, True
        idx = min(idx, len(path) - 1)
        remaining = float(speed)
        while remaining > 1e-6 and idx < len(path) - 1:
            target = path[idx + 1]
            diff = target - pos
            dist = float(np.linalg.norm(diff))
            if dist <= remaining:
                pos = target.copy()
                remaining -= dist
                idx += 1
            else:
                pos += diff / (dist + 1e-8) * remaining
                remaining = 0.0
        done = idx >= len(path) - 1 and np.linalg.norm(pos - path[-1]) < 2.0
        return pos, idx, done

    def _formation_move_step(self):
        """A* 경로를 따라 세 드론을 삼각대형 꼭짓점으로 이동."""
        if self.mode != 'formation_move':
            return
        all_done = True
        for i in range(3):
            p, idx, done = self._move_along_path(
                self.formation_drone_positions[i],
                self.formation_paths[i],
                self.formation_path_indices[i],
            )
            self.formation_drone_positions[i] = p
            self.formation_path_indices[i] = idx
            all_done = all_done and done

        if all_done:
            self.final_drone_positions = self.formation_drone_positions.copy()
            self.sim_result = 'Formation Complete'
            print('=' * 60)
            print('[Formation] 삼각대형 완성 → Localization 실행')
            print(f'  final drone positions:\n{np.round(self.final_drone_positions, 2)}')
            print('=' * 60)
            ok = self._run_localization()
            if ok:
                self.mode = 'done'
                self.sim_done = True
                self.sim_result = 'Localization Done'
            else:
                # Localization 계산 실패 시 DONE으로 덮어쓰지 않는다.
                # 오른쪽 패널에 오류 메시지를 보여주고 사용자가 원인을 확인할 수 있게 유지한다.
                self.mode = 'done'
                self.sim_done = True

    # =================================================================
    #  시뮬레이션 렌더링
    # =================================================================
    def _render_simulation(self):
        self.screen.fill(DARK_BG)

        # 맵 배경 (장애물) — 뷰포트 오프셋 적용, 바깥 영역은 흰색
        self.screen.fill((255, 255, 255), (0, 0, MAP_PX, MAP_PX))   # 맵 영역 흰색
        self.screen.blit(self._sim_bg, (int(self._voff[0]), int(self._voff[1])))

        # 격자 구분선
        for i in range(1, 3):
            v = MAP_PX * i // 3
            pygame.draw.line(self.screen, GRID_COL, (v, 0), (v, MAP_PX), 1)
            pygame.draw.line(self.screen, GRID_COL, (0, v), (MAP_PX, v), 1)

        # BPSK 신호 범위 (반투명 원)
        bsx, bsy = self._vs(*self.env.bpsk_pos)
        overlay = pygame.Surface((MAP_PX, MAP_PX), pygame.SRCALPHA)
        pygame.draw.circle(overlay, (*BPSK_COL, 28), (bsx, bsy), BPSK_SIGNAL_RADIUS)
        self.screen.blit(overlay, (0, 0))
        pygame.draw.circle(self.screen, BPSK_COL, (bsx, bsy), BPSK_SIGNAL_RADIUS, 1)
        pygame.draw.circle(self.screen, BPSK_COL, (bsx, bsy), 9)

        # QAM 방해꾼
        for qp in _QAM_POS:
            qsx, qsy = e2s(qp[0], qp[1])
            qoverlay = pygame.Surface((MAP_PX, MAP_PX), pygame.SRCALPHA)
            pygame.draw.circle(qoverlay, (*QAM_COL, 18), (qsx, qsy), QAM_SIGNAL_RADIUS)
            self.screen.blit(qoverlay, (0, 0))
            pygame.draw.circle(self.screen, QAM_COL, (qsx, qsy), 6)

        # 드론 궤적
        traj = np.array(self.trajectory)
        if len(traj) > 1:
            pts = [self._vs(x, y) for x, y in traj]
            pygame.draw.lines(self.screen, TRAJ_COL, False, pts, 2)

        # 라이다 광선은 PPO 이동/대형 이동 단계에서만 표시.
        # Localization 결과 화면에서는 삼변측량 원/추정점이 잘 보이도록 숨김.
        if getattr(self, 'localization_result', None) is None and not str(getattr(self, 'sim_result', '')).startswith('Localization'):
            dpos = self.env.drone_pos
            dsx, dsy = self._vs(*dpos)
            angles = np.linspace(0, 2 * np.pi, NUM_LIDAR_RAYS, endpoint=False)
            lidar_surf = pygame.Surface((MAP_PX, MAP_PX), pygame.SRCALPHA)
            for angle, nd in zip(angles, self._lidar):
                ray_len = nd * LIDAR_MAX_RANGE
                ex = dpos[0] + np.cos(angle) * ray_len
                ey = dpos[1] + np.sin(angle) * ray_len
                esx, esy = self._vs(ex, ey)
                rc = int(255 * (1.0 - nd))
                gc = int(200 * nd)
                pygame.draw.line(lidar_surf, (rc, gc, 40, 130), (dsx, dsy), (esx, esy), 1)
            self.screen.blit(lidar_surf, (0, 0))

        # 시간차 추종 표시: 리더가 먼저 가고, 팔로워 2대가 같은 경로를 차례대로 따라감
        formation_positions = self._get_formation_positions()
        formation_screen_pts = [self._vs(pos[0], pos[1]) for pos in formation_positions]

        # 이동 단계에 따라 연결선 표시: 추종 중에는 일렬, 삼각대형/결과 단계에서는 삼각형
        if len(formation_screen_pts) == 3:
            if self.mode == 'formation_move' or getattr(self, 'localization_result', None) is not None:
                pygame.draw.polygon(self.screen, FORMATION_LINE_COL, formation_screen_pts, 1)
            else:
                pygame.draw.line(self.screen, FORMATION_LINE_COL,
                                 formation_screen_pts[0], formation_screen_pts[1], 1)
                pygame.draw.line(self.screen, FORMATION_LINE_COL,
                                 formation_screen_pts[1], formation_screen_pts[2], 1)

        # A* 경로 미리보기
        if self.mode == 'formation_move' and getattr(self, 'formation_paths', None) is not None:
            for path in self.formation_paths:
                if path is not None and len(path) > 1:
                    path_pts = [self._vs(p[0], p[1]) for p in path]
                    pygame.draw.lines(self.screen, (255, 240, 150), False, path_pts, 1)
        if self.mode == 'formation_move' and getattr(self, 'formation_targets', None) is not None:
            for t in self.formation_targets:
                tx, ty = self._vs(t[0], t[1])
                pygame.draw.circle(self.screen, (255, 235, 80), (tx, ty), 7, 2)

        # 작은 원 3개: 0번은 리더, 1~2번은 리더 경로를 시간차로 따라오는 팔로워
        drone_draw_info = [
            (formation_screen_pts[0], DRONE_COL,      'D1'),
            (formation_screen_pts[1], FOLLOWER_COL_1, 'D2'),
            (formation_screen_pts[2], FOLLOWER_COL_2, 'D3'),
        ]
        for point, color, label in drone_draw_info:
            pygame.draw.circle(self.screen, color, point, 5)
            pygame.draw.circle(self.screen, WHITE, point, 5, 1)
            self.screen.blit(self.fn_s.render(label, True, color),
                             (point[0] + 6, point[1] - 8))


        # Localization 결과 표시: 추정 원 + 추정 좌표
        if getattr(self, 'localization_result', None) is not None:
            self._render_localization_overlay()

        # ── 오른쪽 패널 ──────────────────────────────────────────────
        self._render_sim_panel()
        pygame.display.flip()


    def _render_localization_overlay(self):
        """Localization 버튼 클릭 후 지도 위에 삼변측량 결과를 표시."""
        res = self.localization_result
        if res is None:
            return

        # 1) 각 드론 기준 거리 원 표시
        #    Kalman 단독 버전은 사용하지 않으므로 파란 Kalman 원은 표시하지 않는다.
        #    Baseline 원(빨강), Kalman+ANN 원(초록)만 표시한다.
        drone_positions = res.get('drone_positions')
        if drone_positions is not None:
            for i, p in enumerate(drone_positions):
                sx, sy = self._vs(p[0], p[1])

                if 'circle_base_px' in res:
                    pygame.draw.circle(
                        self.screen,
                        LOC_CIRCLE_BASE,
                        (sx, sy),
                        int(res['circle_base_px'][i]),
                        1
                    )

                if 'circle_ann_px' in res:
                    pygame.draw.circle(
                        self.screen,
                        LOC_CIRCLE_ANN,
                        (sx, sy),
                        int(res['circle_ann_px'][i]),
                        3
                    )

        # 2) 실제 목표와 추정점 표시
        def draw_plus(pos, color, size=10, width=3):
            x, y = self._vs(pos[0], pos[1])
            pygame.draw.line(self.screen, color, (x - size, y), (x + size, y), width)
            pygame.draw.line(self.screen, color, (x, y - size), (x, y + size), width)
            return x, y

        def draw_square(pos, color, size=8, width=0):
            x, y = self._vs(pos[0], pos[1])
            rect = pygame.Rect(x - size, y - size, size * 2, size * 2)
            pygame.draw.rect(self.screen, color, rect, width)
            return x, y

        true_col = LOC_KAL_COL

        # True: 파란색
        if 'true_target' in res:
            tx, ty = self._vs(res['true_target'][0], res['true_target'][1])
            pygame.draw.circle(self.screen, true_col, (tx, ty), 11)
            self.screen.blit(self.fn_s.render('True', True, true_col), (tx + 12, ty - 8))

        # Baseline 예상 지점: 빨간 네모
        if 'estimated_baseline' in res:
            x, y = draw_square(res['estimated_baseline'], LOC_BASE_COL, size=8, width=0)
            self.screen.blit(self.fn_s.render('Base', True, LOC_BASE_COL), (x + 10, y - 8))

        # Kalman+ANN 예상 지점: 초록 십자가(+)
        if 'estimated_ann' in res:
            x, y = draw_plus(res['estimated_ann'], LOC_ANN_COL, size=10, width=4)
            self.screen.blit(self.fn_s.render('ANN', True, LOC_ANN_COL), (x + 12, y - 8))

    def _render_sim_panel(self):
        px = MAP_PX
        pygame.draw.rect(self.screen, PANEL_BG, (px, 0, PANEL_W, WIN_H))
        pygame.draw.line(self.screen, (65, 70, 90), (px, 0), (px, WIN_H), 2)

        y = 8

        # 제목
        self.screen.blit(self.fn_xl.render('Simulation', True, TITLE_COL), (px + 12, y)); y += 36

        # 모델 파일명
        self.screen.blit(
            self.fn_s.render(f'model: {self._model_label}', True, DIM_COL),
            (px + 12, y)); y += 20

        # 상태 뱃지
        if   self.mode == 'running':              s_col, s_txt = WARN_COL, '● RUNNING'
        elif self.mode == 'followers_catchup':    s_col, s_txt = WARN_COL, '● FOLLOWERS CATCHUP'
        elif self.mode == 'ready_localization':   s_col, s_txt = OK_COL,   '✓ ALL DRONES ARRIVED'
        elif self.mode == 'formation_move':       s_col, s_txt = WARN_COL, '● FORMATION MOVE'
        elif self.sim_result == 'Localization Done': s_col, s_txt = OK_COL, '✓ LOCALIZATION DONE'
        elif 'Crash' in self.sim_result:          s_col, s_txt = ERR_COL,  f'✕ {self.sim_result}'
        else:                                     s_col, s_txt = TEXT_COL, '◌ TIMEOUT'
        self.screen.blit(self.fn_l.render(s_txt, True, s_col), (px + 12, y)); y += 32

        pygame.draw.line(self.screen, (58, 63, 82), (px+8, y), (px+PANEL_W-8, y)); y += 10

        # 오른쪽 패널의 공통 행 출력 함수
        def row(label, val, col=TEXT_COL):
            nonlocal y
            self.screen.blit(self.fn_m.render(label, True, DIM_COL),  (px + 14, y))
            self.screen.blit(self.fn_m.render(str(val), True, col),   (px + 148, y))
            y += 22

        def small_row(label, val, col=TEXT_COL):
            nonlocal y
            self.screen.blit(self.fn_s.render(label, True, DIM_COL),  (px + 14, y))
            self.screen.blit(self.fn_s.render(str(val), True, col),   (px + 148, y))
            y += 18

        def draw_error_bar_chart(base_err, ann_err):
            """Steps/Reward 대신 Baseline vs Kalman+ANN 오차 막대그래프 표시."""
            nonlocal y
            title = 'Error Comparison'
            self.screen.blit(self.fn_m.render(title, True, TITLE_COL), (px + 14, y)); y += 22

            chart_x = px + 28
            chart_y = y
            chart_w = PANEL_W - 56
            chart_h = 100
            axis_y = chart_y + chart_h - 20

            pygame.draw.rect(self.screen, (31, 35, 46), (chart_x, chart_y, chart_w, chart_h), border_radius=4)
            pygame.draw.line(self.screen, (85, 90, 108), (chart_x + 8, axis_y), (chart_x + chart_w - 8, axis_y), 1)

            max_err = max(float(base_err), float(ann_err), 1.0)
            max_err *= 1.20
            bar_w = 54
            gap = 56
            base_x = chart_x + 45
            ann_x = base_x + bar_w + gap
            usable_h = chart_h - 44

            bars = [
                ('Baseline', float(base_err), LOC_BASE_COL, base_x),
                ('Kal+ANN',  float(ann_err),  LOC_ANN_COL,  ann_x),
            ]
            for name, value, color, bx in bars:
                bh = int((value / max_err) * usable_h)
                by = axis_y - bh
                pygame.draw.rect(self.screen, color, (bx, by, bar_w, bh), border_radius=3)
                pygame.draw.rect(self.screen, (90, 98, 120), (bx, by, bar_w, bh), 1, border_radius=3)

                value_txt = self.fn_s.render(f'{value:.2f}m', True, color)
                self.screen.blit(value_txt, value_txt.get_rect(center=(bx + bar_w // 2, by - 8)))
                label_txt = self.fn_s.render(name, True, TEXT_COL)
                self.screen.blit(label_txt, label_txt.get_rect(center=(bx + bar_w // 2, axis_y + 12)))

            y += chart_h + 8

        # Localization 결과가 있으면 오른쪽 패널은 PPO step/reward 대신
        # 오차 비교 그래프, 링크 상태, 결과 해석, 거리 추정, RSSI 샘플을 표시한다.
        if str(getattr(self, 'sim_result', '')).startswith('Localization Error'):
            self.screen.blit(self.fn_m.render('Localization Error', True, ERR_COL), (px + 14, y)); y += 24
            msg = str(self.sim_result).replace('Localization Error:', '').strip()
            # 긴 오류 메시지는 패널 폭에 맞춰 대략 잘라서 여러 줄로 표시
            chunks = [msg[i:i+34] for i in range(0, len(msg), 34)] or ['Unknown error']
            for line in chunks[:8]:
                self.screen.blit(self.fn_s.render(line, True, ERR_COL), (px + 14, y)); y += 18

        elif getattr(self, 'localization_result', None) is not None:
            res = self.localization_result
            labels = res.get('link_labels', ['-', '-', '-'])
            all_los = all(str(lab).upper() == 'LOS' for lab in labels)

            # 1) 상단: Baseline vs Kalman+ANN 오차 막대그래프
            draw_error_bar_chart(res['err_baseline_m'], res['err_ann_m'])

            pygame.draw.line(self.screen, (58, 63, 82), (px+8, y), (px+PANEL_W-8, y)); y += 8

            # 2) Link Status: 맵의 D1/D2/D3 라벨과 이름 일치
            self.screen.blit(self.fn_m.render('Link Status', True, TITLE_COL), (px + 14, y)); y += 22
            for i, lab in enumerate(labels):
                row(f'D{i+1} Link:', lab)

            y += 4
            # 3) Localization Result: Baseline과 Kalman+ANN만 표시
            self.screen.blit(self.fn_m.render('Localization Result', True, TITLE_COL), (px + 14, y)); y += 24
            row('Base Err   :', f"{res['err_baseline_m']:.2f} m", LOC_BASE_COL)
            row('Kal+ANN Err:', f"{res['err_ann_m']:.2f} m", LOC_ANN_COL)

            y += 4
            # 4) Result Analysis: LOS 여부에 따른 추천 결과 표시
            self.screen.blit(self.fn_m.render('Result Analysis', True, TITLE_COL), (px + 14, y)); y += 22
            if all_los:
                analysis_lines = [
                    '링크 3개 모두 LOS 상태로',
                    'Baseline 결과가 더 적합합니다.'
                ]
                analysis_col = LOC_BASE_COL
            else:
                analysis_lines = [
                    '링크가 장애물의 영향을 받아',
                    '오차가 발생하므로 Kal+ANN',
                    '보정 결과가 더 적합합니다.'
                ]
                analysis_col = LOC_ANN_COL
            for line in analysis_lines:
                self.screen.blit(self.fn_kr_s.render(line, True, analysis_col), (px + 14, y)); y += 18

            y += 4
            # 5) Distance Estimate: 추천 방식에 해당하는 거리값만 표시
            self.screen.blit(self.fn_m.render('Distance Estimate', True, TITLE_COL), (px + 14, y)); y += 22
            if 'd_true_m' in res:
                row('True d:', ', '.join(f"{v:.1f}" for v in res['d_true_m']))
            if all_los:
                if 'd_base_m' in res:
                    row('Base d:', ', '.join(f"{v:.1f}" for v in res['d_base_m']), LOC_BASE_COL)
            else:
                if 'd_ann_m' in res:
                    row('Kal+ANN d:', ', '.join(f"{v:.1f}" for v in res['d_ann_m']), LOC_ANN_COL)

            y += 4
            # 6) 마지막: RSSI 원본/칼만/표준편차 샘플 표시
            self.screen.blit(self.fn_m.render('RSSI Samples', True, TITLE_COL), (px + 14, y)); y += 22
            if 'rssi_raw' in res:
                small_row('Raw:', ', '.join(f"{v:.1f}" for v in res['rssi_raw']))
            if 'rssi_kalman' in res:
                small_row('Kal:', ', '.join(f"{v:.1f}" for v in res['rssi_kalman']))
            if 'rssi_std' in res:
                small_row('Std:', ', '.join(f"{v:.1f}" for v in res['rssi_std']))

        else:
            # ── Belief 3×3 히트맵 ────────────────────────────────────────
            self.screen.blit(self.fn_m.render('Belief Map', True, TITLE_COL), (px + 14, y)); y += 22
            cell = 90
            bx = px + (PANEL_W - cell * 3) // 2
            belief = self.env.belief
            for r_i in range(3):
                for c_i in range(3):
                    idx  = (2 - r_i) * 3 + c_i
                    prob = float(belief[idx])
                    iv   = int(prob * 255)
                    rect = pygame.Rect(bx + c_i * cell, y + r_i * cell, cell-3, cell-3)
                    pygame.draw.rect(self.screen, (iv//4, iv//2, iv), rect, border_radius=5)
                    pygame.draw.rect(self.screen, (80, 85, 105), rect, 1, border_radius=5)
                    self.screen.blit(
                        self.fn_s.render(f'{prob*100:.0f}%', True, WHITE),
                        self.fn_s.render(f'{prob*100:.0f}%', True, WHITE).get_rect(center=rect.center))
            y += cell * 3 + 8

            # ── 라이다 레이더 미니맵 ──────────────────────────────────────
            pygame.draw.line(self.screen, (58, 63, 82), (px+8, y), (px+PANEL_W-8, y)); y += 8
            self.screen.blit(self.fn_m.render('LiDAR', True, TITLE_COL), (px + 14, y)); y += 20
            radar_cx = px + PANEL_W // 2
            radar_cy = y + 55
            radar_r  = 52
            pygame.draw.circle(self.screen, (50, 55, 70), (radar_cx, radar_cy), radar_r)
            pygame.draw.circle(self.screen, (70, 75, 95), (radar_cx, radar_cy), radar_r, 1)
            pygame.draw.circle(self.screen, (58, 63, 82), (radar_cx, radar_cy), radar_r//2, 1)

            angles = np.linspace(0, 2 * np.pi, NUM_LIDAR_RAYS, endpoint=False)
            for angle, nd in zip(angles, self._lidar):
                rlen = nd * radar_r
                ex = radar_cx + int(np.cos(angle)  * rlen)
                ey = radar_cy - int(np.sin(angle)  * rlen)
                rc = int(255 * (1.0 - nd))
                gc = int(200 * nd)
                pygame.draw.line(self.screen, (rc, gc, 40), (radar_cx, radar_cy), (ex, ey), 1)
                pygame.draw.circle(self.screen, (rc, gc, 40), (ex, ey), 2)
            pygame.draw.circle(self.screen, DRONE_COL, (radar_cx, radar_cy), 4)

        # Localization 버튼: 클릭하면 A* 삼각대형 이동을 시작하고, 완료 후 삼변측량 자동 실행
        if self.mode == 'ready_localization':
            self.btn_localization.draw(self.screen, self.fn_m)

        # 뒤로 버튼
        self.btn_back.draw(self.screen, self.fn_m)

    # =================================================================
    #  Localization 호출
    # =================================================================
    def _run_localization(self):
        """삼각대형 완성 후 실제 삼변측량 모듈 호출. 성공하면 True, 실패하면 False 반환."""
        if self.final_drone_positions is None:
            self.final_drone_positions = np.array(self._get_formation_positions(), dtype=float)

        print(f"[Localization] drone_positions=\n{self.final_drone_positions}")

        try:
            result = run_localization(
                occupancy_map    = self.env.occupancy.copy(),
                sdf_map          = self.env.sdf_map.copy(),
                target_pos       = self.env.bpsk_pos.copy(),
                drone_positions  = self.final_drone_positions.copy(),
                base_dir         = os.path.dirname(os.path.abspath(__file__)),
            )
        except Exception as e:
            print("[Localization Error]", repr(e))
            self.sim_result = f"Localization Error: {e}"
            self.localization_result = None
            return False

        self.localization_result = result
        self.sim_result = 'Localization Done'
        return True

    # =================================================================
    #  메인 루프
    # =================================================================
    def run(self):
        while True:
            events = pygame.event.get()
            if self.mode in ('running', 'followers_catchup', 'ready_localization', 'formation_move', 'done'):
                self._sim_events(events)
                if self.mode == 'running':
                    self._sim_step()
                elif self.mode == 'followers_catchup':
                    self._followers_catchup_step()
                elif self.mode == 'formation_move':
                    self._formation_move_step()
                self._render_simulation()
            else:
                self._editor_events(events)
                self._render_editor()
            self.clock.tick(30)


if __name__ == '__main__':
    app = App()
    app.run()