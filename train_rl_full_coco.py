#!/usr/bin/env python
"""
Train RL agent on FULL COCO bbob-constrained suite for better generalization.

Key differences from train_rl_enhanced.py:
1. Uses ALL 54 bbob-constrained functions (instead of just 6)
2. Uses more instances (1-5 instead of 1-3)
3. Longer training (20k episodes)
4. Better diversity for generalization

This should produce agents that generalize better to unseen problems.

Setup:
- Suite: bbob-constrained (all functions)
- Functions: ALL available (typically 54 functions)
- Instances: [1, 2, 3, 4, 5]
- Dimensions: [2, 10]

For each dimension d:
- Episodes: 20_000 (increased for diversity)
- Initial design: n_init = 4 * d random points
- Horizon: H = 15 RL steps per episode

The trained policies are saved as:
  models/coco_policy_full_dim2.pt
  models/coco_policy_full_dim10.pt
"""

import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import cocoex
from gymnasium import spaces


# -----------------------------------------------------------------------------
# GLOBAL CONFIG
# -----------------------------------------------------------------------------

# Use ALL available COCO functions (let COCO determine what's available)
# We'll sample from the full suite
INSTANCES = [1, 2, 3, 4, 5]  # More instances for diversity
DIMENSIONS = [2, 10]

N_EPISODES = 20_000  # Increased for more diverse training
HORIZON = 15
N_INITIAL_FACTOR = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32

# Reproducibility
GLOBAL_SEED = 123
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)


# -----------------------------------------------------------------------------
# PPO COMPONENTS (same as enhanced)
# -----------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """Gaussian policy."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(state)
        mean = self.mean_layer(h)
        std = torch.exp(self.log_std)
        return mean, std


class ValueNetwork(nn.Module):
    """State-value function V(s)."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.value = ValueNetwork(state_dim, hidden_dim).to(DEVICE)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        state_t = torch.tensor(state, dtype=TORCH_DTYPE, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.policy(state_t)
        if deterministic:
            action = mean
            log_prob = torch.zeros_like(action).sum(dim=-1)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        # Map from R^d to [0,1]^d via tanh + rescale
        action = torch.tanh(action)
        action = (action + 1.0) / 2.0
        action = action.clamp(0.0, 1.0)

        return action.cpu().numpy()[0].astype(np.float32), float(log_prob.cpu().item())

    def update(self, trajectories: List[Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, float]]):
        if len(trajectories) == 0:
            return

        states = []
        actions = []
        old_log_probs = []
        returns = []

        for s, a, lp, r, ns, d in trajectories:
            states.append(s)
            actions.append(a)
            old_log_probs.append(lp)

            state_t = torch.tensor(s, dtype=TORCH_DTYPE, device=DEVICE).unsqueeze(0)
            next_state_t = torch.tensor(ns, dtype=TORCH_DTYPE, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                v_s = self.value(state_t).item()
                v_ns = self.value(next_state_t).item()

            target = r + self.gamma * v_ns * (1.0 - float(d))
            returns.append(target)

        states_t = torch.tensor(np.array(states), dtype=TORCH_DTYPE, device=DEVICE)
        actions_t = torch.tensor(np.array(actions), dtype=TORCH_DTYPE, device=DEVICE)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=TORCH_DTYPE, device=DEVICE)
        returns_t = torch.tensor(returns, dtype=TORCH_DTYPE, device=DEVICE)

        # PPO update
        mean, std = self.policy(states_t)
        actions_raw = 2.0 * actions_t - 1.0
        actions_raw = actions_raw.clamp(-0.999, 0.999)
        actions_raw = torch.atanh(actions_raw)

        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        values = self.value(states_t)
        advantages = returns_t - values.detach()

        ratio = torch.exp(log_probs - old_log_probs_t)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = ((returns_t - values) ** 2).mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value.parameters()), 0.5
        )
        self.optimizer.step()


# -----------------------------------------------------------------------------
# FULL COCO ENVIRONMENT
# -----------------------------------------------------------------------------

class FullCocoConstrainedBOEnv:
    """
    Environment that samples from ENTIRE COCO bbob-constrained suite.
    """

    def __init__(
        self,
        dim: int,
        horizon: int = 15,
        n_initial: int = None,
    ):
        self.dim = dim
        self.horizon = horizon
        self.n_initial = n_initial or (4 * dim)

        self.suite = cocoex.Suite("bbob-constrained", "", "")

        # Get ALL available problems for this dimension
        self.all_problems = []
        for problem in self.suite:
            if problem.dimension == dim:
                # Store (function_id, instance)
                fid = problem.id.split('_')[1][1:]  # Extract function number
                inst_str = problem.id.split('_')[2][1:]  # Extract instance number
                try:
                    fid_num = int(fid)
                    inst_num = int(inst_str)
                    if inst_num in INSTANCES:
                        self.all_problems.append((fid_num, inst_num))
                except ValueError:
                    pass

        # Remove duplicates
        self.all_problems = list(set(self.all_problems))

        print(f"[INFO] Found {len(self.all_problems)} problems for dim={dim}")
        print(f"       Functions: {sorted(set(f for f, _ in self.all_problems))}")

        # Enhanced state dimension (same as enhanced version)
        self.state_dim = 16 + 3 + dim + dim

        # Internal state
        self.current_problem = None
        self.X_observed = []
        self.y_observed = []
        self.c_observed = []
        self.steps_taken = 0
        self.best_feasible_value = -np.inf

    def _sample_problem(self):
        """Sample uniformly from all available problems."""
        fid, inst = random.choice(self.all_problems)
        self.current_problem = self.suite.get_problem_by_function_dimension_instance(
            fid, self.dim, inst
        )

    def _evaluate(self, x_norm: np.ndarray) -> Tuple[float, float]:
        lb = np.array(self.current_problem.lower_bounds, dtype=np.float32)
        ub = np.array(self.current_problem.upper_bounds, dtype=np.float32)
        x_real = lb + x_norm * (ub - lb)

        f_val = self.current_problem(x_real)
        y = -float(f_val)

        if hasattr(self.current_problem, "constraint"):
            g_vals = np.asarray(self.current_problem.constraint(x_real), dtype=np.float32)
            c_agg = float(g_vals.max()) if g_vals.size > 0 else 0.0
        else:
            c_agg = 0.0

        return y, c_agg

    def _add_observation(self, x: np.ndarray, y: float, c: float):
        self.X_observed.append(x.copy())
        self.y_observed.append(y)
        self.c_observed.append(c)

        if c <= 0.0 and y > self.best_feasible_value:
            self.best_feasible_value = y

    def _local_surrogate_features(self) -> np.ndarray:
        n = len(self.X_observed)
        if n < 5:
            return np.zeros(3, dtype=np.float32)

        X = np.array(self.X_observed, dtype=np.float32)
        y = np.array(self.y_observed, dtype=np.float32)

        MAX_POINTS = 50
        if n > MAX_POINTS:
            X = X[-MAX_POINTS:]
            y = y[-MAX_POINTS:]

        MAX_CENTERS = min(5, len(X))
        centers = X[-MAX_CENTERS:]

        sigma = 0.2 * np.sqrt(self.dim)
        if sigma <= 0.0:
            return np.zeros(3, dtype=np.float32)
        denom = 2.0 * (sigma ** 2)

        def rbf_kernel(A, B):
            dists_sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
            return np.exp(-dists_sq / denom)

        K = rbf_kernel(X, centers)
        lam = 1e-3
        KT_K = K.T @ K
        KT_y = K.T @ y
        A_mat = KT_K + lam * np.eye(MAX_CENTERS, dtype=np.float32)
        try:
            alpha = np.linalg.solve(A_mat, KT_y)
        except np.linalg.LinAlgError:
            return np.zeros(3, dtype=np.float32)

        y_hat = K @ alpha
        residuals = y - y_hat
        residual_std = float(np.std(residuals))

        last_x = X[-1:]
        y_arr = np.array(self.y_observed, dtype=np.float32)
        c_arr = np.array(self.c_observed, dtype=np.float32)
        feasible_mask = c_arr <= 0.0
        if feasible_mask.any():
            best_idx = int(np.argmax(y_arr * feasible_mask))
            best_x = np.array(self.X_observed[best_idx], dtype=np.float32)[None, :]
        else:
            best_x = last_x.copy()

        K_last = rbf_kernel(last_x, centers)
        K_best = rbf_kernel(best_x, centers)

        mu_last = float((K_last @ alpha).ravel()[0])
        mu_best = float((K_best @ alpha).ravel()[0])

        return np.array([mu_last, mu_best, residual_std], dtype=np.float32)

    def _get_state(self) -> np.ndarray:
        if len(self.X_observed) == 0:
            return np.zeros(self.state_dim, dtype=np.float32)

        y_arr = np.array(self.y_observed, dtype=np.float32)
        c_arr = np.array(self.c_observed, dtype=np.float32)
        feasible_mask = c_arr <= 0.0
        best_feasible = (
            float(np.max(y_arr[feasible_mask])) if feasible_mask.any() else -np.inf
        )

        y_mean = float(np.mean(y_arr))
        y_std = float(np.std(y_arr) + 1e-8)
        y_max = float(np.max(y_arr))
        n_feasible = int(np.sum(feasible_mask))
        feasible_ratio = float(n_feasible) / float(len(c_arr))

        # Enhanced constraint statistics
        c_mean = float(np.mean(c_arr))
        c_std = float(np.std(c_arr) + 1e-8)
        c_min = float(np.min(c_arr))
        last_c = float(c_arr[-1])

        last_point = np.array(self.X_observed[-1], dtype=np.float32)

        if feasible_mask.any():
            best_idx = int(np.argmax(y_arr * feasible_mask))
            best_point = np.array(self.X_observed[best_idx], dtype=np.float32)
            best_c = float(c_arr[best_idx])
        else:
            # Use least infeasible point
            least_infeas_idx = int(np.argmin(c_arr))
            best_point = np.array(self.X_observed[least_infeas_idx], dtype=np.float32)
            best_c = float(c_arr[least_infeas_idx])

        if np.isneginf(best_feasible):
            best_feasible = 0.0

        progress = float(self.steps_taken) / float(self.horizon)

        summary = np.array(
            [
                best_feasible,
                y_mean,
                y_std,
                y_max,
                float(n_feasible),
                feasible_ratio,
                float(len(self.X_observed)),
                float(self.steps_taken),
                float(self.horizon),
                float(self.dim),
                c_mean,
                c_std,
                c_min,
                last_c,
                best_c,
                progress,
            ],
            dtype=np.float32,
        )

        surrogate_feats = self._local_surrogate_features()

        state = np.concatenate(
            [summary, surrogate_feats, last_point, best_point],
            axis=0,
        ).astype(np.float32)

        if state.shape[0] != self.state_dim:
            if state.shape[0] < self.state_dim:
                pad = np.zeros(self.state_dim - state.shape[0], dtype=np.float32)
                state = np.concatenate([state, pad], axis=0)
            else:
                state = state[: self.state_dim]

        return state

    def _compute_reward(self, old_best: float, y: float, c: float, x_norm: np.ndarray) -> float:
        """Balanced reward with log1p scaling."""
        is_feasible = c <= 0.0

        # Novelty bonus
        if len(self.X_observed) > 1:
            prev_points = np.array(self.X_observed[:-1], dtype=np.float32)
            dists = np.linalg.norm(prev_points - x_norm[None, :], axis=1)
            min_dist = float(np.min(dists))
            novelty = np.clip(min_dist / np.sqrt(self.dim), 0.0, 1.0)
        else:
            novelty = 1.0

        had_feasible_before = not np.isneginf(old_best)
        reward = 0.0

        if not had_feasible_before:
            if is_feasible:
                reward = 5.0 + 0.5 * novelty
            else:
                violation = max(0.0, c)
                viol_penalty = -1.0 * np.log1p(violation) / 10.0

                if len(self.c_observed) > 1:
                    prev_c_min = min(self.c_observed[:-1])
                    if c < prev_c_min:
                        improvement_ratio = (prev_c_min - c) / (abs(prev_c_min) + 1.0)
                        approach_bonus = 1.0 * np.clip(improvement_ratio, 0.0, 1.0)
                    else:
                        approach_bonus = 0.0
                else:
                    approach_bonus = 0.0

                reward = viol_penalty + approach_bonus + 0.2 * novelty
        else:
            if is_feasible:
                improvement = max(0.0, self.best_feasible_value - old_best)
                denom = abs(old_best) + 1.0
                norm_improvement = improvement / denom
                reward = 3.0 * norm_improvement + 0.2 * novelty
            else:
                violation = max(0.0, c)
                viol_penalty = -0.5 * np.log1p(violation) / 10.0
                reward = viol_penalty + 0.1 * novelty

        reward = float(np.clip(reward, -5.0, 5.0))
        return reward

    def reset(self) -> np.ndarray:
        self._sample_problem()
        self.X_observed = []
        self.y_observed = []
        self.c_observed = []
        self.steps_taken = 0
        self.best_feasible_value = -np.inf

        # Initial random design
        for _ in range(self.n_initial):
            x_norm = np.random.rand(self.dim).astype(np.float32)
            y, c = self._evaluate(x_norm)
            self._add_observation(x_norm, y, c)

        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, 0.0, 1.0)

        old_best = self.best_feasible_value
        y, c = self._evaluate(action)
        self._add_observation(action, y, c)

        self.steps_taken += 1
        done = self.steps_taken >= self.horizon

        reward = self._compute_reward(old_best, y, c, action)
        state = self._get_state()
        info: Dict[str, Any] = {"y": y, "c": c}
        return state, reward, done, info


# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------

def train_dimension(dim: int):
    print(f"\n{'='*70}")
    print(f"Training FULL COCO Policy for dim={dim}")
    print(f"{'='*70}\n")

    env = FullCocoConstrainedBOEnv(dim=dim, horizon=HORIZON)
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=dim,
        hidden_dim=256,
        lr=3e-4,
    )

    episode_rewards = []
    feasibility_rates = []

    for ep in range(N_EPISODES):
        state = env.reset()
        trajectories = []
        episode_reward = 0.0
        n_feasible = 0

        for _ in range(HORIZON):
            action, log_prob = agent.select_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            trajectories.append((state, action, log_prob, reward, next_state, done))
            episode_reward += reward
            if info['c'] <= 0.0:
                n_feasible += 1
            state = next_state
            if done:
                break

        agent.update(trajectories)
        episode_rewards.append(episode_reward)
        feasibility_rates.append(n_feasible / HORIZON)

        if (ep + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            avg_feas = np.mean(feasibility_rates[-1000:])
            print(f"Episode {ep+1}/{N_EPISODES}, "
                  f"Avg Reward: {avg_reward:6.2f}, "
                  f"Feasibility: {avg_feas:.1%}")

    # Save checkpoint
    os.makedirs("models", exist_ok=True)
    ckpt_path = f"models/coco_policy_full_dim{dim}.pt"
    torch.save(
        {
            "policy_state_dict": agent.policy.state_dict(),
            "value_state_dict": agent.value.state_dict(),
            "dim": dim,
            "horizon": HORIZON,
            "state_dim": env.state_dim,
            "n_problems_trained": len(env.all_problems),
        },
        ckpt_path,
    )
    print(f"\nSaved FULL COCO policy to {ckpt_path}")
    print(f"State dim: {env.state_dim}")
    print(f"Trained on {len(env.all_problems)} unique problems")


if __name__ == "__main__":
    for dim in DIMENSIONS:
        train_dimension(dim)
    print("\n" + "="*70)
    print("FULL COCO Training Complete!")
    print("="*70)
    print("\nThese policies should generalize better due to:")
    print("  - Training on ALL COCO functions (not just 6)")
    print("  - More instances (5 instead of 3)")
    print("  - Longer training (20k episodes)")
