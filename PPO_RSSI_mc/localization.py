"""
localization.py

PPO + LiDAR 시뮬레이터에서 Localization 버튼을 눌렀을 때 호출되는 모듈.
- 입력 좌표계: pygame/env 좌표계 700 x 700 px
- 내부 RSSI/ANN/삼변측량 좌표계: 기존 Localization 코드의 0~100 m 좌표계
- 출력 좌표계: 다시 700 x 700 px로 변환해서 pygame에서 바로 표시 가능
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np

from ann_train import MODEL_FILE, load_model
from kalman_filter import kalman_filter
from rssi_env import (
    K_SAMPLES,
    PARTIAL_NLOS_RADIUS,
    RealDataChannel,
    dist_from_rssi_baseline,
    trilat_2d_ls,
)


MAP_METERS = 100.0  # 기존 localization 알고리즘의 좌표계 크기


# link_type: 0=LOS, 1=Partial NLOS, 2=Full NLOS
LINK_LABELS = {0: "LOS", 1: "PNLOS", 2: "NLOS"}


def _px_to_m(pos_px: np.ndarray, map_size_px: float) -> np.ndarray:
    """700px env 좌표를 0~100m localization 좌표로 변환."""
    return np.asarray(pos_px, dtype=float) * (MAP_METERS / float(map_size_px))


def _m_to_px(pos_m: np.ndarray, map_size_px: float) -> np.ndarray:
    """0~100m localization 좌표를 700px env 좌표로 변환."""
    return np.asarray(pos_m, dtype=float) * (float(map_size_px) / MAP_METERS)


def _line_blocked_by_occupancy(occupancy_map: np.ndarray, a_px: np.ndarray, b_px: np.ndarray) -> bool:
    """두 점 사이 직선 경로가 장애물 픽셀을 통과하면 Full NLOS로 판단."""
    h, w = occupancy_map.shape
    a_px = np.asarray(a_px, dtype=float)
    b_px = np.asarray(b_px, dtype=float)
    dist = float(np.linalg.norm(b_px - a_px))
    steps = max(2, int(dist / 3.0))
    ts = np.linspace(0.0, 1.0, steps)
    pts = a_px + ts[:, None] * (b_px - a_px)
    xs = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
    return bool(np.any(occupancy_map[ys, xs]))


def _nearest_obstacle_distance_m(sdf_map: np.ndarray | None, pos_px: np.ndarray, map_size_px: float) -> float:
    """sdf_map의 px 단위 장애물 거리값을 m 단위로 변환."""
    if sdf_map is None:
        return np.inf
    h, w = sdf_map.shape
    x = int(np.clip(round(float(pos_px[0])), 0, w - 1))
    y = int(np.clip(round(float(pos_px[1])), 0, h - 1))
    return float(sdf_map[y, x]) * (MAP_METERS / float(map_size_px))


def _sample_real_rssi_with_map(
    channel: RealDataChannel,
    drone_px: np.ndarray,
    target_px: np.ndarray,
    occupancy_map: np.ndarray,
    sdf_map: np.ndarray | None,
    map_size_px: float,
    rng: np.random.Generator,
    K: int = K_SAMPLES,
    partial_r_m: float = PARTIAL_NLOS_RADIUS,
) -> Tuple[float, float, float, int]:
    """
    pygame의 실제 장애물 맵을 기준으로 LOS/PNLOS/NLOS를 판정하고,
    실측 RSSI 데이터셋에서 거리별 RSSI 샘플을 가져온다.
    """
    drone_m = _px_to_m(drone_px, map_size_px)
    target_m = _px_to_m(target_px, map_size_px)
    d_m = float(np.linalg.norm(drone_m - target_m))

    # Full NLOS: 경로 차단
    if _line_blocked_by_occupancy(occupancy_map, drone_px, target_px):
        raw = channel._draw(max(d_m, 5.0), channel.nlos_pools, channel.nlos_d, K)
        return float(raw.mean()), float(kalman_filter(raw)), float(raw.std()), 2

    # Partial NLOS: 드론 근처 장애물
    obs_dist_m = _nearest_obstacle_distance_m(sdf_map, drone_px, map_size_px)
    if obs_dist_m < partial_r_m:
        half = K // 2
        raw_los  = rng.choice(
            channel._draw(max(d_m, 1.0), channel.los_pools,  channel.los_d,  K), half, replace=True)
        raw_nlos = rng.choice(
            channel._draw(max(d_m, 5.0), channel.nlos_pools, channel.nlos_d, K), half, replace=True)
        raw = np.concatenate([raw_los, raw_nlos])
        rng.shuffle(raw)
        return float(raw.mean()), float(kalman_filter(raw)), float(raw.std()), 1

    # LOS
    raw = channel._draw(max(d_m, 1.0), channel.los_pools, channel.los_d, K)
    return float(raw.mean()), float(kalman_filter(raw)), float(raw.std()), 0


def run_localization(
    occupancy_map: np.ndarray,
    sdf_map: np.ndarray,
    target_pos: np.ndarray,
    drone_positions: np.ndarray,
    base_dir: str | Path | None = None,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    occupancy_map : np.ndarray, shape=(700,700), bool
        True = 장애물
    sdf_map : np.ndarray, shape=(700,700), float
        장애물까지 거리(px)
    target_pos : np.ndarray, shape=(2,)
        실제 목표 위치(px). 시뮬레이션 RSSI 생성을 위해 사용.
    drone_positions : np.ndarray, shape=(3,2)
        리더, F1, F2 최종 위치(px)
    base_dir : str | Path
        ann_model.npz, rssi_no_obstacle.xlsx, rssi_with_obstacle.xlsx가 있는 폴더.

    Returns
    -------
    dict
        pygame에서 바로 렌더링 가능한 결과. 좌표는 px, 거리 원 반지름은 px.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    else:
        base_dir = Path(base_dir)

    rng = np.random.default_rng(int(time.time()) if seed is None else seed)

    no_obs_path = base_dir / "rssi_no_obstacle.xlsx"
    with_obs_path = base_dir / "rssi_with_obstacle.xlsx"
    # 모델은 다른 모델과 같은 위치(실행 위치=프로젝트 루트)에서 찾는다.
    # cwd에 없으면 코드 폴더(base_dir)도 시도(하위호환).
    model_path = Path(MODEL_FILE)
    if not model_path.exists():
        model_path = base_dir / MODEL_FILE

    missing = [str(p) for p in [no_obs_path, with_obs_path, model_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Localization에 필요한 파일이 없습니다:\n" + "\n".join(missing))

    drone_px = np.asarray(drone_positions, dtype=float)
    target_px = np.asarray(target_pos, dtype=float)
    map_size_px = float(max(occupancy_map.shape))

    if drone_px.shape != (3, 2):
        raise ValueError(f"drone_positions는 shape=(3,2)여야 합니다. 현재: {drone_px.shape}")

    channel = RealDataChannel(no_obs_path, with_obs_path, rng)
    model, mu, sd = load_model(model_path)

    raw_means, kal_means, stds, link_types = [], [], [], []
    for k in range(3):
        rm, km, st, lt = _sample_real_rssi_with_map(
            channel=channel,
            drone_px=drone_px[k],
            target_px=target_px,
            occupancy_map=occupancy_map,
            sdf_map=sdf_map,
            map_size_px=map_size_px,
            rng=rng,
            K=K_SAMPLES,
            partial_r_m=PARTIAL_NLOS_RADIUS,
        )
        raw_means.append(rm)
        kal_means.append(km)
        stds.append(st)
        link_types.append(lt)

    rssi_raw = np.array(raw_means, dtype=float)
    rssi_kal = np.array(kal_means, dtype=float)
    rssi_std = np.array(stds, dtype=float)
    link_types = np.array(link_types, dtype=int)

    # 삼변측량은 기존 localization 알고리즘 기준인 0~100m 좌표계에서 수행
    A_m = _px_to_m(drone_px, map_size_px)
    target_m = _px_to_m(target_px, map_size_px)
    d_true_m = np.linalg.norm(A_m - target_m[None, :], axis=1)

    # 1) Baseline: raw mean RSSI -> distance -> trilateration
    d_base_m = dist_from_rssi_baseline(rssi_raw)
    x_base_m = trilat_2d_ls(A_m, d_base_m)

    # 2) Kalman: Kalman RSSI -> distance -> trilateration
    d_kal_m = dist_from_rssi_baseline(rssi_kal)
    x_kal_m = trilat_2d_ls(A_m, d_kal_m)

    # 3) Kalman + ANN: ANN estimates distances
    #    입력 = [칼만RSSI 3] + [link_type one-hot 9].  std 대신 link 상태를 직접 제공.
    #    link_types(0=LOS,1=PNLOS,2=NLOS)는 위에서 맵 기반으로 이미 판정됨.
    onehot = np.zeros(9, dtype=float)
    for k in range(3):
        onehot[k * 3 + int(link_types[k])] = 1.0
    feat = np.concatenate([rssi_kal, onehot])
    z = (feat - mu) / sd
    d_ann_m, _ = model.forward(z.reshape(1, 12))
    d_ann_m = np.maximum(d_ann_m.reshape(-1), 0.2)
    x_ann_m = trilat_2d_ls(A_m, d_ann_m)

    # 출력은 pygame에 그릴 수 있게 px 좌표/반지름도 같이 반환
    x_base_px = _m_to_px(x_base_m, map_size_px)
    x_kal_px = _m_to_px(x_kal_m, map_size_px)
    x_ann_px = _m_to_px(x_ann_m, map_size_px)

    err_base_m = float(np.linalg.norm(x_base_m - target_m))
    err_kal_m = float(np.linalg.norm(x_kal_m - target_m))
    err_ann_m = float(np.linalg.norm(x_ann_m - target_m))

    scale_px_per_m = map_size_px / MAP_METERS

    result: Dict[str, Any] = {
        "drone_positions": drone_px,
        "true_target": target_px,
        "estimated_baseline": x_base_px,
        "estimated_kalman": x_kal_px,
        "estimated_ann": x_ann_px,
        "err_baseline_m": err_base_m,
        "err_kalman_m": err_kal_m,
        "err_ann_m": err_ann_m,
        "rssi_raw": rssi_raw,
        "rssi_kalman": rssi_kal,
        "rssi_std": rssi_std,
        "link_types": link_types,
        "link_labels": [LINK_LABELS[int(t)] for t in link_types],
        "d_true_m": d_true_m,
        "d_base_m": d_base_m,
        "d_kalman_m": d_kal_m,
        "d_ann_m": d_ann_m,
        "circle_base_px": d_base_m * scale_px_per_m,
        "circle_kalman_px": d_kal_m * scale_px_per_m,
        "circle_ann_px": d_ann_m * scale_px_per_m,
        "scale_px_per_m": scale_px_per_m,
        "map_size_px": map_size_px,
    }

    print("=" * 60)
    print("[Localization] 완료")
    print(f"  Link types      : {result['link_labels']}")
    print(f"  True dist(m)    : {np.round(d_true_m, 2)}")
    print(f"  Raw mean(dBm)   : {np.round(rssi_raw, 2)}")
    print(f"  Kalman(dBm)     : {np.round(rssi_kal, 2)}")
    print(f"  RSSI std(dBm)   : {np.round(rssi_std, 2)}")
    print(f"  Err Baseline(m) : {err_base_m:.3f}")
    print(f"  Err Kalman(m)   : {err_kal_m:.3f}")
    print(f"  Err Kal+ANN(m)  : {err_ann_m:.3f}")
    print("=" * 60)

    return result