import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Actor-Critic 신경망 (공유 레이어 없이 Actor / Critic 완전 분리)

    ── Actor ──────────────────────────────────────────────────────
    입력(state_dim) → Linear(256) → ReLU → Linear(256) → ReLU
                   → Linear(action_dim) → Tanh
    출력: 행동 평균(mu) ∈ [-1, 1]

    log_std: -0.5 고정 (std ≈ 0.6)
    ─ 기존에는 학습 가능한 파라미터였기 때문에
      entropy_coef 감소 + log_std 감소가 동시에 일어나는
      "이중 탐험 억제" 문제가 있었음
    ─ 고정함으로써 탐험량을 entropy_coef 하나로만 제어,
      스케줄이 예측 가능하고 깔끔해짐
    ─ std=0.6은 행동 다양성을 충분히 유지하는 값

    ── Critic ─────────────────────────────────────────────────────
    입력(state_dim) → Linear(256) → ReLU → Linear(256) → ReLU
                   → Linear(1)
    출력: 상태 가치 V(s) (스칼라)
    """

    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor 네트워크: 상태 → 연속 행동 평균
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()          # 출력 범위를 [-1, 1] 로 제한
        )

        # [수정] log_std 고정 (requires_grad=False)
        # -0.5로 초기화 → std = exp(-0.5) ≈ 0.606
        # entropy_coef(train.py 코사인 스케줄)만으로 탐험량 단일 제어
        self.log_std = nn.Parameter(
            torch.full((1, action_dim), -0.5),
            requires_grad=False
        )

        # Critic 네트워크: 상태 → 가치 추정
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        """
        반환값
        -------
        dist  : Normal 분포 객체 (행동 샘플링 및 log_prob 계산에 사용)
        value : 상태 가치 V(s), shape = [batch, 1]
        """
        value  = self.critic(state)
        mu     = self.actor(state)
        std    = self.log_std.exp().expand_as(mu)
        dist   = Normal(mu, std)
        return dist, value


class PPO:
    """
    Proximal Policy Optimization (PPO-Clip) 구현

    핵심 수식
    ---------
    ratio     = π_θ(a|s) / π_θ_old(a|s)   ← exp(log_prob - old_log_prob)
    surr1     = ratio × advantage
    surr2     = clip(ratio, 1-ε, 1+ε) × advantage
    L_actor   = -min(surr1, surr2)          ← 정책 개선 손실
    L_critic  = MSE(V(s), R_t)             ← 가치 추정 손실
    L_entropy = -H(π)                       ← 탐험 장려 (보너스)
    L_total   = L_actor + 0.5×L_critic - entropy_coef×L_entropy

    하이퍼파라미터
    -------------
    lr            : 1e-4  (Adam 학습률)
    gamma         : 0.99  (할인율)
    eps_clip      : 0.1   (클리핑 범위)
    k_epochs      : 10    (같은 데이터로 반복 학습 횟수)
    batch_size    : 1024  (미니배치 크기)
    max_grad_norm : 0.5   (그래디언트 클리핑 임계값)
    """

    def __init__(self, state_dim, action_dim,
                 lr=1e-4, gamma=0.99, eps_clip=0.1):
        self.gamma        = gamma
        self.eps_clip     = eps_clip
        self.k_epochs     = 10
        self.batch_size   = 1024

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ PPO Device Set to: {self.device}")

        self.policy    = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss  = nn.MSELoss()
        self.reward_window = collections.deque(maxlen=50)

    # ──────────────────────────────────────────────────────────────
    def select_action(self, state):
        """
        학습 중 행동 선택 (탐험 포함)
        Normal 분포에서 샘플링 → 확률적 탐험
        """
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            dist, value = self.policy(state)
            action   = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy().flatten()

    # ──────────────────────────────────────────────────────────────
    def save_if_best(self, current_reward, path="best_model.pth"):
        self.reward_window.append(current_reward)
        if current_reward >= max(self.reward_window):
            torch.save(self.policy.state_dict(), path)
            print(f"✨ New Best Saved: {current_reward:.2f}")

    def update(self, memory, entropy_coef):
        """
        수집된 experience로 정책 업데이트

        1. Discounted return G_t 계산 (역방향 누적)
        2. return 정규화 (학습 안정화)
        3. Advantage = G_t - V(s) 계산 후 정규화
        4. k_epochs 반복 × 미니배치 셔플 → PPO 손실 역전파
        """
        # ── Tensor 변환 ───────────────────────────────────────────
        states       = torch.FloatTensor(np.array(memory['states'])).to(self.device)
        actions      = torch.FloatTensor(np.array(memory['actions'])).to(self.device)
        old_log_probs= torch.FloatTensor(np.array(memory['log_probs'])).to(self.device)

        rewards      = np.array(memory['rewards'])
        dones        = np.array(memory['dones'])
        values       = np.array(memory['values'])

        # ── 1. Discounted Return 계산 ─────────────────────────────
        returns           = []
        discounted_reward = np.zeros(rewards.shape[1])
        for reward, done in zip(reversed(rewards), reversed(dones)):
            discounted_reward = reward + (self.gamma * discounted_reward * (1 - done))
            returns.insert(0, discounted_reward)

        # ── 2. Return 정규화 ──────────────────────────────────────
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # ── 3. Advantage 계산 + 정규화 ────────────────────────────
        values     = torch.FloatTensor(values).to(self.device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # ── Flatten ───────────────────────────────────────────────
        b_states     = states.view(-1, states.shape[-1])
        b_actions    = actions.view(-1, actions.shape[-1])
        b_old_lp     = old_log_probs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns    = returns.view(-1)

        # ── 4. 미니배치 PPO 업데이트 ──────────────────────────────
        dataset_size = b_states.shape[0]
        indices      = np.arange(dataset_size)

        for _ in range(self.k_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end       = start + self.batch_size
                batch_idx = indices[start:end]

                mb_states     = b_states[batch_idx]
                mb_actions    = b_actions[batch_idx]
                mb_old_lp     = b_old_lp[batch_idx]
                mb_advantages = b_advantages[batch_idx]
                mb_returns    = b_returns[batch_idx]

                dist, state_values = self.policy(mb_states)
                log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy   = dist.entropy().sum(dim=-1)

                # PPO-Clip 손실
                ratios = torch.exp(log_probs - mb_old_lp)
                surr1  = ratios * mb_advantages
                surr2  = torch.clamp(ratios,
                                     1 - self.eps_clip,
                                     1 + self.eps_clip) * mb_advantages

                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss =  self.mse_loss(state_values.squeeze(), mb_returns)

                # 총 손실: Actor + 0.5×Critic - entropy_coef×Entropy
                loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()