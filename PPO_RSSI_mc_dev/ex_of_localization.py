"""
localization.py  —  Localization 모듈 스텁

run_localization() 은 탐색 완료 후 시뮬레이터에서 호출.

Parameters (모두 시뮬레이터가 자동으로 전달)
----------
occupancy_map : np.ndarray, shape (700, 700), dtype=bool
    True = 장애물 / 벽, False = 빈 공간
    좌표계: occupancy[y, x], y=0 이 맵 아래쪽

sdf_map : np.ndarray, shape (700, 700), dtype=float32
    각 픽셀에서 가장 가까운 장애물까지의 거리(픽셀 단위)
    빈 공간이면 양수, 장애물 내부이면 0

target_pos : np.ndarray, shape (2,), dtype=float64
    BPSK 신호원(타겟) 위치 [x, y]  (env 좌표계)

drone_pos : np.ndarray, shape (2,), dtype=float64
    탐색 완료 시점의 드론 위치 [x, y]  (env 좌표계)

Returns
원하는 거 담아서 return 예를 들어 드론 3개 위치, 신호원 위치, 원 등등.


[개발 가이드]
localization.py
  run_localization() 안에서
  모든 계산 수행
      └─ 결과 dict 반환
           (추정 위치, 신뢰원 반지름, 후보 좌표들 등)

pygame_simul.py
  _run_localization() 에서 결과 받아서
  self.mode = 'localization' 으로 전환
      └─ _render_localization() 에서
         결과값만 가지고 그림만 그리기
이렇게 구현할 예정.

"""




import numpy as np


def run_localization(occupancy_map, sdf_map, target_pos, drone_pos):
    """
    Localization 알고리즘 진입점.
    현재는 입력값을 그대로 반환하는 스텁입니다.
    """

    print("=" * 55)
    print("[Localization] 호출됨")
    print(f"  occupancy_map  : shape={occupancy_map.shape}, "
          f"obstacles={occupancy_map.sum()} px")
    print(f"  sdf_map        : shape={sdf_map.shape}, "
          f"min={sdf_map.min():.1f}, max={sdf_map.max():.1f}")
    print(f"  target_pos     : {target_pos}")
    print(f"  drone_pos      : {drone_pos}")
    print("=" * 55)

    # ── TODO: 여기에 실제 localization 로직을 구현하세요 ────────────

    result = {
        'estimated_pos':    drone_pos.copy(),   # 추정 드론 위치 (스텁: 그대로)
        'estimated_target': target_pos.copy(),  # 추정 타겟 위치 (스텁: 그대로)
        'confidence':       1.0,                # 신뢰도 (스텁: 1.0)
        # 필요 시 추가 키를 여기에 넣으세요
    }

    print(f"[Localization] 결과: {result}")
    return result