import math
import time
from pathlib import Path

import numpy as np

from rssi_env import K_SAMPLES, RealDataChannel, make_one_scenario


MODEL_FILE = "ann_model.npz"


def relu(x):
    return np.maximum(x, 0.0)


def relu_grad(x):
    return (x > 0.0).astype(float)


class MLP6to3:
    """
    Input:  [Kalman RSSI x3, link_type one-hot x9]  (드론3 × [LOS,PNLOS,NLOS])
    Output: [distance to drone1, distance to drone2, distance to drone3]

    표준편차(std) 입력을 제거하고 link_type one-hot으로 대체했다.
    이 데이터셋은 LOS/NLOS의 std 차이가 거의 없어 std가 NLOS 힌트가 되지 못한다.
    대신 rssi_env/localization이 이미 판정하는 link_type(LOS/PNLOS/NLOS)을
    one-hot으로 직접 넣어, 모델이 상황을 확실히 알고 상황별로 보정하게 한다.
    (클래스명은 호환을 위해 유지하나 실제 입력 차원은 12다.)
    """

    def __init__(self, hidden, rng):
        self.H = hidden
        self.D = 12
        self.dim = hidden * 12 + hidden + 3 * hidden + 3
        self.w = 0.05 * rng.standard_normal(self.dim)

    def unpack(self, w):
        H = self.H
        D = self.D
        i = 0

        W1 = w[i:i + H * 12].reshape(H, 12)
        i += H * 12

        b1 = w[i:i + H]
        i += H

        W2 = w[i:i + 3 * H].reshape(3, H)
        i += 3 * H

        b2 = w[i:i + 3]

        return W1, b1, W2, b2

    def forward(self, X, w=None):
        if w is None:
            w = self.w

        W1, b1, W2, b2 = self.unpack(w)

        Z1 = X @ W1.T + b1
        A1 = relu(Z1)
        Y = A1 @ W2.T + b2

        return Y, (X, Z1, A1, W2)

    def mse_grad(self, X, Ygt, w=None):
        if w is None:
            w = self.w

        Yp, (X0, Z1, A1, W2) = self.forward(X, w)

        diff = Yp - Ygt
        N = X.shape[0]
        mse = float(np.mean(diff ** 2))

        dY = 2 / (N * 3) * diff

        W1, b1, _, _ = self.unpack(w)

        dW2 = dY.T @ A1
        db2 = dY.sum(0)

        dA1 = dY @ W2
        dZ1 = dA1 * relu_grad(Z1)

        dW1 = dZ1.T @ X0
        db1 = dZ1.sum(0)

        g = np.zeros_like(w)
        i = 0
        H = self.H

        g[i:i + H * 12] = dW1.reshape(-1)
        i += H * 12

        g[i:i + H] = db1
        i += H

        g[i:i + 3 * H] = dW2.reshape(-1)
        i += 3 * H

        g[i:i + 3] = db2

        return mse, g


def train_ann(
    rng,
    channel,
    K=K_SAMPLES,
    Ntr=12000,
    Nva=2000,
    hidden=40,
    epochs=400,
    batch=256,
    lr0=8e-3,
    momentum=0.9,
    wd=1e-5,
):
    def _onehot(lt):
        v = [0.0, 0.0, 0.0]
        v[int(lt)] = 1.0      # 0=LOS, 1=PNLOS, 2=NLOS
        return v

    def one_sample():
        x_true, A = make_one_scenario(rng)
        d_true = np.linalg.norm(A - x_true[None, :], axis=1)

        kal_means = []
        onehots = []

        for k in range(3):
            _, kal, std, lt = channel.sample(A[k], x_true, K)
            kal_means.append(kal)
            onehots.extend(_onehot(lt))   # 드론별 [LOS,PNLOS,NLOS] 3비트

        # 입력: [칼만RSSI 3개] + [link one-hot 9개] = 12
        feat = np.array(kal_means + onehots)
        return feat, d_true

    Xtr = np.zeros((Ntr, 12))
    Ytr = np.zeros((Ntr, 3))
    Xva = np.zeros((Nva, 12))
    Yva = np.zeros((Nva, 3))

    for i in range(Ntr):
        Xtr[i], Ytr[i] = one_sample()

    for i in range(Nva):
        Xva[i], Yva[i] = one_sample()

    # 정규화는 RSSI(앞 3개)만. one-hot(뒤 9개)은 0/1이라 정규화하지 않는다.
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    mu[0, 3:] = 0.0   # one-hot 부분은 평균 0
    sd[0, 3:] = 1.0   # one-hot 부분은 스케일 1 (그대로 통과)

    Ztr = (Xtr - mu) / sd
    Zva = (Xva - mu) / sd

    model = MLP6to3(hidden=hidden, rng=rng)

    v = np.zeros_like(model.w)
    best_mse = np.inf
    best_w = model.w.copy()

    for ep in range(1, epochs + 1):
        lr = lr0 * (1 + math.cos(math.pi * ep / epochs)) / 2
        perm = rng.permutation(Ntr)

        for s in range(0, Ntr, batch):
            idx = perm[s:s + batch]
            _, grad = model.mse_grad(Ztr[idx], Ytr[idx])
            grad += wd * model.w

            v = momentum * v + (1 - momentum) * grad
            model.w -= lr * v

        if ep % 50 == 0:
            yv, _ = model.forward(Zva)
            mse = float(np.mean((yv - Yva) ** 2))

            if mse < best_mse:
                best_mse = mse
                best_w = model.w.copy()

            print(f"  [Epoch {ep:03d}] val MSE={mse:.4f}  best={best_mse:.4f}")

    model.w = best_w
    return model, mu.reshape(-1), sd.reshape(-1), best_mse


def save_model(path, model, mu, sd, best_mse):
    np.savez(
        path,
        weights=model.w,
        mu=mu,
        sd=sd,
        hidden=np.array([model.H], dtype=int),
        best_mse=np.array([best_mse], dtype=float),
    )


def load_model(path):
    data = np.load(path)
    hidden = int(data["hidden"][0])

    rng = np.random.default_rng(0)
    model = MLP6to3(hidden=hidden, rng=rng)

    model.w = data["weights"]
    mu = data["mu"]
    sd = data["sd"]

    return model, mu, sd


def main():
    base_dir = Path(__file__).resolve().parent

    no_obs_path = base_dir / "rssi_no_obstacle.xlsx"
    with_obs_path = base_dir / "rssi_with_obstacle.xlsx"
    # 모델은 다른 모델(ppo 등)과 같은 위치(실행 위치=프로젝트 루트)에 저장.
    # 엑셀은 코드 옆(base_dir)에서 찾되, 모델 파일만 cwd에 둔다.
    model_path = Path(MODEL_FILE)   # 파일명만 → 실행한 위치(cwd)에 생성

    if not no_obs_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {no_obs_path}")

    if not with_obs_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {with_obs_path}")

    rng = np.random.default_rng(int(time.time()))
    channel = RealDataChannel(no_obs_path, with_obs_path, rng)

    print("ANN 학습 시작")
    print("입력: Kalman RSSI 3개 + env_index 9개")
    print("출력: 드론 3대와 목표물 사이의 거리 3개")

    model, mu, sd, best_mse = train_ann(rng, channel)

    save_model(model_path, model, mu, sd, best_mse)

    print(f"\nANN 모델 저장 완료: {model_path}")
    print(f"Best validation MSE: {best_mse:.4f}")
    print("\n이제부터는 test.py만 실행하면 저장된 ANN 모델을 불러와 테스트합니다.")


if __name__ == "__main__":
    main()