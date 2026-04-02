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
    batch_size    : 64    (미니배치 크기) -> [멀티코어 수정: 1024로 증가]
    max_grad_norm : 0.5   (그래디언트 클리핑 임계값)
    """

    def __init__(self, state_dim, action_dim,
                 lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.gamma        = gamma
        self.eps_clip     = eps_clip
        self.k_epochs     = 10
        self.batch_size   = 1024 # [멀티코어 수정: GPU 활용을 위해 64에서 1024로 대폭 증가]

        # [멀티코어 수정: CUDA(GPU)가 있으면 자동 할당, 없으면 CPU 사용]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ PPO Device Set to: {self.device}")

        self.policy    = ActorCritic(state_dim, action_dim).to(self.device) # [멀티코어 수정: 신경망을 Device로 이동]
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss  = nn.MSELoss()
        self.reward_window = collections.deque(maxlen=50)
    # ──────────────────────────────────────────────────────────────
    def select_action(self, state):
        """
        학습 중 행동 선택 (탐험 포함)
        Normal 분포에서 샘플링 → 확률적 탐험
        """
        # [멀티코어 수정: 여러 대의 드론 데이터를 한 번에 처리하므로 unsqueeze(0) 제거 및 device 할당]
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            dist, value = self.policy(state)
            action   = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        # [멀티코어 수정: GPU에서 계산된 텐서를 다시 CPU numpy 배열로 변환하여 환경에 전달]
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy().flatten()

    # ──────────────────────────────────────────────────────────────
    def save_if_best(self, current_reward, path="best_model.pth"):
        self.reward_window.append(current_reward)
        if current_reward >= max(self.reward_window):
            torch.save(self.policy.state_dict(), path)
            print(f"✨ New Best Saved: {current_reward:.2f}")
            
    def update(self, memory):
        """
        수집된 experience로 정책 업데이트

        1. Discounted return G_t 계산 (역방향 누적)
        2. return 정규화 (학습 안정화)
        3. Advantage = G_t - V(s) 계산 후 정규화
        4. k_epochs 반복 × 미니배치 셔플 → PPO 손실 역전파
        """
        # ── Tensor 변환 ───────────────────────────────────────────
        # [멀티코어 수정: 2차원 배열(스텝×드론)을 device로 올림]
        states       = torch.FloatTensor(np.array(memory['states'])).to(self.device)
        actions      = torch.FloatTensor(np.array(memory['actions'])).to(self.device)
        old_log_probs= torch.FloatTensor(np.array(memory['log_probs'])).to(self.device)
        
        rewards      = np.array(memory['rewards'])
        dones        = np.array(memory['dones'])
        values       = np.array(memory['values'])

        # ── 1. Discounted Return 계산 ─────────────────────────────
        # 에피소드가 끝나면(done=True) 누적 보상 리셋
        returns          = []
        # [멀티코어 수정: 드론 32대 분량의 보상을 동시에 누적하도록 배열 연산으로 변경]
        discounted_reward = np.zeros(rewards.shape[1]) 
        for reward, done in zip(reversed(rewards), reversed(dones)):
            discounted_reward = reward + (self.gamma * discounted_reward * (1 - done)) # [멀티코어 수정: done이 1이면 누적 보상 리셋]
            returns.insert(0, discounted_reward)

        # ── 2. Return 정규화 ──────────────────────────────────────
        returns = torch.FloatTensor(np.array(returns)).to(self.device) # [멀티코어 수정: device 할당]
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # ── 3. Advantage 계산 + 정규화 ────────────────────────────
        values     = torch.FloatTensor(values).to(self.device) # [멀티코어 수정: device 할당]
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # [멀티코어 수정: 여러 대의 드론이 수집한 데이터를 1차원으로 쫙 펴기 (Flatten) - 학습을 위해 섞기 위함]
        b_states     = states.view(-1, states.shape[-1])
        b_actions    = actions.view(-1, actions.shape[-1])
        b_old_lp     = old_log_probs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns    = returns.view(-1)

        # ── 4. 미니배치 PPO 업데이트 ──────────────────────────────
        dataset_size = b_states.shape[0] # [멀티코어 수정: 평탄화된 데이터 크기로 변경]
        indices      = np.arange(dataset_size)

        for _ in range(self.k_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end       = start + self.batch_size
                batch_idx = indices[start:end]

                mb_states     = b_states[batch_idx]     # [멀티코어 수정: b_states 등 평탄화된 변수명 사용]
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

                # 총 손실: Actor + 0.5×Critic - 0.01×Entropy
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                # 그래디언트 폭주 방지 (norm 0.5 이내로 클리핑)
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()