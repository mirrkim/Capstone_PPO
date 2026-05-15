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
import pygame
import numpy as np
import torch
import scipy.ndimage

from env import (DroneEnv, MAP_SIZE,
                 BPSK_SIGNAL_RADIUS, QAM_SIGNAL_RADIUS,
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
TRAJ_COL    = (220, 50,  180)
BPSK_COL    = (210, 60,  60)
QAM_COL     = (60,  110, 220)
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
            qsx, qsy = e2s(qp[0], qp[1])
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
    #  시뮬레이션 시작
    # =================================================================
    def _launch(self):
        # ── 모델 로드 ────────────────────────────────────────────────
        best_path  = 'ppo_drone_best.pth'
        final_path = 'ppo_drone_final.pth'
        if   os.path.exists(best_path):  load_path = best_path
        elif os.path.exists(final_path): load_path = final_path
        else:                            load_path = None

        self.env = DroneEnv()

        # ── env.py 무수정으로 커스텀 맵 주입 ────────────────────────
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
        self._lidar         = np.ones(NUM_LIDAR_RAYS, dtype=np.float32)

        # 맵 배경 서피스 1회 생성 (시뮬레이션 중 불변)
        self._sim_bg = self._get_map_surf(occ=self.env.occupancy)

        self.mode = 'running'

        # 뒤로 버튼
        self.btn_back = Button((MAP_PX + 12, WIN_H - 58, 292, 46), '◀  Back to Editor')

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

    # =================================================================
    #  시뮬레이션 렌더링
    # =================================================================
    def _render_simulation(self):
        self.screen.fill(DARK_BG)

        # 맵 배경 (장애물)
        self.screen.blit(self._sim_bg, (0, 0))

        # 격자 구분선
        for i in range(1, 3):
            v = MAP_PX * i // 3
            pygame.draw.line(self.screen, GRID_COL, (v, 0), (v, MAP_PX), 1)
            pygame.draw.line(self.screen, GRID_COL, (0, v), (MAP_PX, v), 1)

        # BPSK 신호 범위 (반투명 원)
        bsx, bsy = e2s(*self.env.bpsk_pos)
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
            pts = [e2s(x, y) for x, y in traj]
            pygame.draw.lines(self.screen, TRAJ_COL, False, pts, 2)

        # 라이다 광선 (한 서피스에 전부 그리기 — 속도 최적화)
        dpos = self.env.drone_pos
        dsx, dsy = e2s(*dpos)
        angles = np.linspace(0, 2 * np.pi, NUM_LIDAR_RAYS, endpoint=False)
        lidar_surf = pygame.Surface((MAP_PX, MAP_PX), pygame.SRCALPHA)
        for angle, nd in zip(angles, self._lidar):
            ray_len = nd * LIDAR_MAX_RANGE
            ex = dpos[0] + np.cos(angle) * ray_len
            ey = dpos[1] + np.sin(angle) * ray_len
            esx, esy = e2s(ex, ey)
            # 가까울수록 붉게, 멀수록 노랗게
            rc = int(255 * (1.0 - nd))
            gc = int(200 * nd)
            pygame.draw.line(lidar_surf, (rc, gc, 40, 130), (dsx, dsy), (esx, esy), 1)
        self.screen.blit(lidar_surf, (0, 0))

        # 드론 (현재 위치)
        pygame.draw.circle(self.screen, DRONE_COL, (dsx, dsy), 8)
        pygame.draw.circle(self.screen, WHITE,     (dsx, dsy), 8, 2)

        # ── 오른쪽 패널 ──────────────────────────────────────────────
        self._render_sim_panel()
        pygame.display.flip()

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
        elif self.sim_result == 'Target Found':   s_col, s_txt = OK_COL,   '✓ TARGET FOUND'
        elif 'Crash' in self.sim_result:          s_col, s_txt = ERR_COL,  f'✕ {self.sim_result}'
        else:                                     s_col, s_txt = TEXT_COL, '◌ TIMEOUT'
        self.screen.blit(self.fn_l.render(s_txt, True, s_col), (px + 12, y)); y += 32

        pygame.draw.line(self.screen, (58, 63, 82), (px+8, y), (px+PANEL_W-8, y)); y += 10

        # 수치 정보
        def row(label, val, col=TEXT_COL):
            nonlocal y
            self.screen.blit(self.fn_m.render(label, True, DIM_COL),  (px + 14, y))
            self.screen.blit(self.fn_m.render(str(val), True, col),   (px + 148, y))
            y += 22

        row('Steps  :', f'{self.step_count} / 512')
        row('Step R :', f'{self.last_reward:+.1f}',
            OK_COL if self.last_reward >= 0 else ERR_COL)
        row('Total R:', f'{self.total_reward:+.1f}',
            OK_COL if self.total_reward >= 0 else ERR_COL)

        y += 6
        pygame.draw.line(self.screen, (58, 63, 82), (px+8, y), (px+PANEL_W-8, y)); y += 10

        # ── Belief 3×3 히트맵 ────────────────────────────────────────
        self.screen.blit(self.fn_m.render('Belief Map', True, TITLE_COL), (px + 14, y)); y += 22
        cell = 90
        bx = px + (PANEL_W - cell * 3) // 2
        belief = self.env.belief
        for r_i in range(3):
            for c_i in range(3):
                # row 0 = 맵 아래쪽 → 패널에서는 위아래 반전 (row 2 위에 표시)
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
            # 화면 좌표계에서 y 반전
            ex = radar_cx + int(np.cos(angle)  * rlen)
            ey = radar_cy - int(np.sin(angle)  * rlen)
            rc = int(255 * (1.0 - nd))
            gc = int(200 * nd)
            pygame.draw.line(self.screen, (rc, gc, 40), (radar_cx, radar_cy), (ex, ey), 1)
            pygame.draw.circle(self.screen, (rc, gc, 40), (ex, ey), 2)
        pygame.draw.circle(self.screen, DRONE_COL, (radar_cx, radar_cy), 4)

        # 뒤로 버튼
        self.btn_back.draw(self.screen, self.fn_m)

    # =================================================================
    #  메인 루프
    # =================================================================
    def run(self):
        while True:
            events = pygame.event.get()
            if self.mode in ('running', 'done'):
                self._sim_events(events)
                if self.mode == 'running':
                    self._sim_step()
                self._render_simulation()
            else:
                self._editor_events(events)
                self._render_editor()
            self.clock.tick(30)


if __name__ == '__main__':
    app = App()
    app.run()