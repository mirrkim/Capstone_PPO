import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Actor-Critic 신경망 (공유 레이어 없이 Actor / Critic 완전 분리)

    log_std: -0.5 고정 (std ≈ 0.6) — entropy_coef 하나로만 탐험량 제어
    """

    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )

        self.log_std = nn.Parameter(
            torch.full((1, action_dim), -0.5),
            requires_grad=False
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        mu    = self.actor(state)
        std   = self.log_std.exp().expand_as(mu)
        return Normal(mu, std), value


class PPO:
    """
    Proximal Policy Optimization (PPO-Clip)
    """

    def __init__(self, state_dim, action_dim,
                 lr=5e-5, gamma=0.99, eps_clip=0.1):
        self.gamma      = gamma
        self.eps_clip   = eps_clip
        self.k_epochs   = 10
        self.batch_size = 8192

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ PPO Device Set to: {self.device}")

        self.policy    = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss  = nn.MSELoss()
        self.reward_window = collections.deque(maxlen=50)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            dist, value = self.policy(state)
            action   = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy().flatten()

    def save_if_best(self, current_reward, path="best_model.pth"):
        self.reward_window.append(current_reward)
        if current_reward >= max(self.reward_window):
            torch.save(self.policy.state_dict(), path)
            print(f"✨ New Best Saved: {current_reward:.2f}")

    def update(self, memory, entropy_coef):
        """
        수집된 experience로 정책 업데이트

        [최적화] Discounted Return 계산:
          기존: returns.insert(0, ...) — O(n²) Python 리스트 삽입
          변경: np.zeros 배열에 역방향 인덱스 기록 — O(n)
        """
        states        = torch.FloatTensor(np.array(memory['states'])).to(self.device)
        actions       = torch.FloatTensor(np.array(memory['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(memory['log_probs'])).to(self.device)

        rewards = np.array(memory['rewards'])   # (T, num_envs)
        dones   = np.array(memory['dones'])     # (T, num_envs)
        values  = np.array(memory['values'])    # (T, num_envs)

        # ── Discounted Return — O(n) 역방향 배열 기록 ───────────────
        T, E      = rewards.shape
        returns   = np.empty_like(rewards)
        disc_rew  = np.zeros(E, dtype=np.float32)
        for i in range(T - 1, -1, -1):
            disc_rew  = rewards[i] + self.gamma * disc_rew * (1.0 - dones[i])
            returns[i] = disc_rew

        # ── Return 정규화 ────────────────────────────────────────────
        returns_t = torch.FloatTensor(returns).to(self.device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-7)

        # ── Advantage 계산 + 정규화 ──────────────────────────────────
        values_t    = torch.FloatTensor(values).to(self.device)
        advantages  = returns_t - values_t
        advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # ── Flatten ───────────────────────────────────────────────────
        b_states     = states.view(-1, states.shape[-1])
        b_actions    = actions.view(-1, actions.shape[-1])
        b_old_lp     = old_log_probs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns    = returns_t.view(-1)

        # ── 미니배치 PPO 업데이트 ────────────────────────────────────
        dataset_size = b_states.shape[0]
        indices      = np.arange(dataset_size)

        for _ in range(self.k_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                dist, state_values = self.policy(b_states[batch_idx])
                log_probs = dist.log_prob(b_actions[batch_idx]).sum(dim=-1)
                entropy   = dist.entropy().sum(dim=-1)

                ratios = torch.exp(log_probs - b_old_lp[batch_idx])
                surr1  = ratios * b_advantages[batch_idx]
                surr2  = torch.clamp(ratios,
                                     1 - self.eps_clip,
                                     1 + self.eps_clip) * b_advantages[batch_idx]

                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss =  self.mse_loss(state_values.squeeze(), b_returns[batch_idx])
                loss        = actor_loss + 0.5 * critic_loss - entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()