"""
evaluate.py

학습된 PPO 드론 모델을 랜덤 장애물 환경에서 평가하는 스크립트.
- 매 에피소드마다 env.reset()이 _generate_world()를 호출하므로 장애물이 새로 랜덤 생성됨
- 총 N_EPISODES 에피소드를 돌려 성공률(정확도)과 충돌률을 집계
- 결과를 이동평균/누적 그래프로 저장
- CUDA가 있으면 자동 사용

사용법:
    python evaluate.py
    python evaluate.py --episodes 3000 --model ppo_drone_best.pth
    python evaluate.py --stochastic        # 결정론적(mean) 대신 샘플링 행동 사용
"""
import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 디스플레이 없는 환경에서도 그래프 저장 가능
import matplotlib.pyplot as plt

from env import DroneEnv
from ppo import ActorCritic


# 종료 메시지 분류 (env.py의 step 반환 msg 기준)
SUCCESS_MSGS = {"Target Found"}
CRASH_MSGS   = {"Wall Crash", "Obs Crash"}
TIMEOUT_MSGS = {"Timeout", "Normal"}


def moving_average(x, window):
    """단순 이동평균. 앞부분은 가능한 만큼만 평균."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    csum = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


@torch.no_grad()
def select_action_eval(policy, state, device, deterministic=True):
    """평가용 행동 선택.
    deterministic=True면 분포의 평균(mu)을 그대로 사용(탐험 없음) → 정확도 측정에 적합.
    False면 학습 때처럼 샘플링.

    주의: ActorCritic.forward의 log_std는 shape (1, action_dim)이라
    expand_as(mu)가 동작하려면 입력이 2차원(배치)이어야 한다. 단일 환경의 state는
    1차원 (state_dim,)이므로, 배치 차원을 붙여 (1, state_dim)으로 넣고
    결과에서 다시 그 차원을 제거한다."""
    s = torch.FloatTensor(np.asarray(state)).to(device)
    if s.dim() == 1:
        s = s.unsqueeze(0)          # (state_dim,) → (1, state_dim)
    dist, _ = policy(s)
    if deterministic:
        action = dist.mean          # Normal 분포의 평균 = actor의 tanh 출력(mu)
    else:
        action = dist.sample()
    action = action.squeeze(0)      # (1, action_dim) → (action_dim,)
    return action.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="PPO 드론 모델 평가")
    ap.add_argument("--episodes", type=int, default=3000, help="평가 에피소드 수 (기본 3000)")
    ap.add_argument("--model", type=str, default="ppo_drone_best.pth", help="모델 가중치 경로")
    ap.add_argument("--window", type=int, default=100, help="이동평균 윈도우 (기본 100)")
    ap.add_argument("--stochastic", action="store_true",
                    help="결정론적(mean) 대신 샘플링 행동 사용")
    ap.add_argument("--out", type=str, default="eval_result.png", help="그래프 저장 파일명")
    ap.add_argument("--seed", type=int, default=None, help="재현용 시드(선택)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Eval Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {args.model}")

    # ── 환경 & 정책 준비 ──
    env = DroneEnv()
    policy = ActorCritic(env.state_dim, env.action_dim).to(device)
    policy.load_state_dict(torch.load(args.model, map_location=device))
    policy.eval()
    print(f"🚀 모델 로드 완료: {args.model}")
    print(f"   행동 모드: {'샘플링(stochastic)' if args.stochastic else '결정론적(deterministic)'}")
    print(f"   평가 에피소드: {args.episodes}  (매 에피소드 장애물 랜덤 생성)\n")

    # ── 결과 기록 버퍼 ──
    is_success = np.zeros(args.episodes, dtype=np.float32)  # 1=성공
    is_crash   = np.zeros(args.episodes, dtype=np.float32)  # 1=충돌
    is_timeout = np.zeros(args.episodes, dtype=np.float32)  # 1=타임아웃
    ep_rewards = np.zeros(args.episodes, dtype=np.float32)
    ep_steps   = np.zeros(args.episodes, dtype=np.int32)

    deterministic = not args.stochastic

    for ep in range(args.episodes):
        state = env.reset()           # 내부에서 _generate_world() → 장애물 랜덤화
        done = False
        total_r = 0.0
        steps = 0
        msg = "Normal"

        while not done:
            action = select_action_eval(policy, state, device, deterministic)
            state, reward, done, msg = env.step(action)
            total_r += reward
            steps += 1

        if msg in SUCCESS_MSGS:
            is_success[ep] = 1.0
        elif msg in CRASH_MSGS:
            is_crash[ep] = 1.0
        else:
            is_timeout[ep] = 1.0

        ep_rewards[ep] = total_r
        ep_steps[ep]   = steps

        # 진행 로그
        if (ep + 1) % 100 == 0:
            n = ep + 1
            print(f"[{n:5d}/{args.episodes}] "
                  f"성공 {is_success[:n].mean()*100:5.1f}% | "
                  f"충돌 {is_crash[:n].mean()*100:5.1f}% | "
                  f"타임아웃 {is_timeout[:n].mean()*100:5.1f}% | "
                  f"평균스텝 {ep_steps[:n].mean():5.1f}")

    # ── 최종 집계 ──
    succ_rate = is_success.mean() * 100
    crash_rate = is_crash.mean() * 100
    to_rate = is_timeout.mean() * 100
    print("\n" + "=" * 55)
    print(f"평가 완료: {args.episodes} 에피소드")
    print(f"  성공률(정확도) : {succ_rate:6.2f}%  ({int(is_success.sum())}회)")
    print(f"  충돌률         : {crash_rate:6.2f}%  ({int(is_crash.sum())}회)")
    print(f"  타임아웃률     : {to_rate:6.2f}%  ({int(is_timeout.sum())}회)")
    print(f"  평균 보상      : {ep_rewards.mean():8.1f}")
    print(f"  평균 스텝      : {ep_steps.mean():6.1f}")
    print("=" * 55)

    # ── 그래프 ──
    w = args.window
    succ_ma  = moving_average(is_success, w) * 100
    crash_ma = moving_average(is_crash, w) * 100
    x = np.arange(1, args.episodes + 1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # (1) 성공률(정확도)
    axes[0].plot(x, succ_ma, color="#2ca02c", lw=1.8, label=f"Success rate (MA-{w})")
    axes[0].axhline(succ_rate, color="#2ca02c", ls="--", lw=1.0, alpha=0.6,
                    label=f"Overall {succ_rate:.1f}%")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title(f"Evaluation over {args.episodes} random-obstacle episodes  |  model: {os.path.basename(args.model)}")
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right")

    # (2) 충돌률
    axes[1].plot(x, crash_ma, color="#d62728", lw=1.8, label=f"Crash rate (MA-{w})")
    axes[1].axhline(crash_rate, color="#d62728", ls="--", lw=1.0, alpha=0.6,
                    label=f"Overall {crash_rate:.1f}%")
    axes[1].set_ylabel("Crash rate (%)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"📊 그래프 저장: {args.out}")

    # 원자료도 저장(보고서/재플롯용)
    np.savez("eval_raw.npz",
             success=is_success, crash=is_crash, timeout=is_timeout,
             rewards=ep_rewards, steps=ep_steps)
    print("💾 원자료 저장: eval_raw.npz")


if __name__ == "__main__":
    main()