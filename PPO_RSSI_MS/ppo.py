import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Actor-Critic 신경망 (공유 레이어 없이 Actor / Critic 완전 분리)

    ── Actor ──────────────────────────────────────────────────────
    입력(state_dim) → Linear(256) → ReLU → Linear(256) → ReLU
                   → Linear(action_dim) → Tanh
    출력: 행동 평균(mu) ∈ [-1, 1]

    log_std: 학습 가능한 파라미터로 분산(탐험 강도)을 자동 조정
    → std = exp(log_std) 로 변환 후 Normal 분포 생성

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

        # 탐험 노이즈 크기 (log 스케일로 저장 → 항상 양수 보장)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

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
    L_total   = L_actor + 0.5×L_critic - 0.01×L_entropy

    하이퍼파라미터
    -------------
    lr            : 3e-4  (Adam 학습률)
    gamma         : 0.99  (할인율)
    eps_clip      : 0.2   (클리핑 범위)
    k_epochs      : 10    (같은 데이터로 반복 학습 횟수)
    batch_size    : 64    (미니배치 크기)
    max_grad_norm : 0.5   (그래디언트 클리핑 임계값)
    """

    def __init__(self, state_dim, action_dim,
                 lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.gamma        = gamma
        self.eps_clip     = eps_clip
        self.k_epochs     = 10
        self.batch_size   = 64

        self.policy    = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss  = nn.MSELoss()

    # ──────────────────────────────────────────────────────────────
    def select_action(self, state):
        """
        학습 중 행동 선택 (탐험 포함)
        Normal 분포에서 샘플링 → 확률적 탐험
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.policy(state)
            action   = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.numpy()[0], log_prob.item(), value.item()

    # ──────────────────────────────────────────────────────────────
    def update(self, memory):
        """
        수집된 experience로 정책 업데이트

        1. Discounted return G_t 계산 (역방향 누적)
        2. return 정규화 (학습 안정화)
        3. Advantage = G_t - V(s) 계산 후 정규화
        4. k_epochs 반복 × 미니배치 셔플 → PPO 손실 역전파
        """
        # ── Tensor 변환 ───────────────────────────────────────────
        states       = torch.FloatTensor(np.array(memory['states']))
        actions      = torch.FloatTensor(np.array(memory['actions']))
        old_log_probs= torch.FloatTensor(np.array(memory['log_probs']))
        rewards      = memory['rewards']
        dones        = memory['dones']
        values       = memory['values']

        # ── 1. Discounted Return 계산 ─────────────────────────────
        # 에피소드가 끝나면(done=True) 누적 보상 리셋
        returns          = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        # ── 2. Return 정규화 ──────────────────────────────────────
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # ── 3. Advantage 계산 + 정규화 ────────────────────────────
        values     = torch.FloatTensor(values)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # ── 4. 미니배치 PPO 업데이트 ──────────────────────────────
        dataset_size = len(states)
        indices      = np.arange(dataset_size)

        for _ in range(self.k_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end       = start + self.batch_size
                batch_idx = indices[start:end]

                b_states     = states[batch_idx]
                b_actions    = actions[batch_idx]
                b_old_lp     = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns    = returns[batch_idx]

                dist, state_values = self.policy(b_states)
                log_probs = dist.log_prob(b_actions).sum(dim=-1)
                entropy   = dist.entropy().sum(dim=-1)

                # PPO-Clip 손실
                ratios = torch.exp(log_probs - b_old_lp)
                surr1  = ratios * b_advantages
                surr2  = torch.clamp(ratios,
                                     1 - self.eps_clip,
                                     1 + self.eps_clip) * b_advantages

                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss =  self.mse_loss(state_values.squeeze(), b_returns)

                # 총 손실: Actor + 0.5×Critic - 0.01×Entropy
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                # 그래디언트 폭주 방지 (norm 0.5 이내로 클리핑)
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
