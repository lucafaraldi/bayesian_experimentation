#!/usr/bin/env python
"""
Benchmark FULL COCO pre-trained RL policies against qLogEI and Random Search.

This version works with policies trained by train_rl_full_coco.py which have:
- Enhanced state representation (16 summary + 3 surrogate + 2*dim)
- Trained on ALL COCO functions (not just 6)
- Trained on more instances (1-5 instead of 1-3)
- Should generalize better to unseen problems
"""

import os
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn

import cocoex

# BoTorch setup (qLogEI baseline)
from botorch.models import SingleTaskGP, ModelListGP
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll
import warnings
from gpytorch.utils.warnings import NumericalWarning
from botorch.exceptions.warnings import OptimizationWarning

warnings.filterwarnings("ignore", category=NumericalWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Optimization failed in `gen_candidates_scipy`.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Optimization failed on the second try.*")
warnings.filterwarnings("ignore", category=OptimizationWarning)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

FUNCTIONS = [2, 4, 6, 50, 52, 54]
INSTANCES = [1, 2, 3]
DIMENSIONS = [2, 10]
REPETITIONS = 5
BUDGET_FACTOR = 10
N_INIT_FACTOR = 2

MODEL_DIR = Path("models")
MODEL_PATTERN = "coco_policy_full_dim{dim}.pt"

OUT_CSV = Path("results_coco_rl_full_vs_qlogei.csv")
OUT_CSV_SUMMARY = Path("results_coco_rl_full_vs_qlogei_summary.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

# -----------------------------------------------------------------------------
# ENHANCED RL POLICY NETWORK
# -----------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """Must match the PolicyNetwork used in training."""

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


# -----------------------------------------------------------------------------
# ENHANCED RL OPTIMIZER WRAPPER
# -----------------------------------------------------------------------------

class FullCocoRLOptimizer:
    """
    Full COCO RL optimizer (trained on all COCO functions).
    """

    def __init__(self, dim: int, ckpt_path: Path):
        self.dim = dim

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        self.state_dim = int(ckpt.get("state_dim"))
        hidden_dim = int(ckpt["policy_state_dict"]["shared.0.weight"].shape[0])

        self.policy = PolicyNetwork(self.state_dim, dim, hidden_dim=hidden_dim).to(DEVICE)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.policy.eval()

    def _local_surrogate_features(self, X_hist: np.ndarray, f_hist: np.ndarray, c_hist: np.ndarray) -> np.ndarray:
        """RBF regression features."""
        n = len(X_hist)
        if n < 5:
            return np.zeros(3, dtype=np.float32)

        X = np.array(X_hist, dtype=np.float32)
        y = -np.array(f_hist, dtype=np.float32)

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

        last_x = X[-1:, :]

        y_arr = -np.array(f_hist, dtype=np.float32)
        c_arr = np.array(c_hist, dtype=np.float32)
        feasible_mask = c_arr <= 0.0
        if feasible_mask.any():
            best_idx = int(np.argmax(y_arr * feasible_mask))
            best_x = X_hist[best_idx:best_idx+1, :]
        else:
            best_x = last_x.copy()

        K_last = rbf_kernel(last_x, centers)
        K_best = rbf_kernel(best_x, centers)

        mu_last = float((K_last @ alpha).ravel()[0])
        mu_best = float((K_best @ alpha).ravel()[0])

        return np.array([mu_last, mu_best, residual_std], dtype=np.float32)

    def _build_state(
            self,
            X_hist: np.ndarray,
            f_hist: np.ndarray,
            c_hist: np.ndarray,
            step: int,
            budget: int,
    ) -> np.ndarray:
        """
        ENHANCED state representation matching train_rl_enhanced.py.
        """
        if len(X_hist) == 0:
            return np.zeros(self.state_dim, dtype=np.float32)

        y_arr = -np.array(f_hist, dtype=np.float32)
        c_arr = np.array(c_hist, dtype=np.float32)

        feasible_mask = c_arr <= 0.0
        if feasible_mask.any():
            best_feasible = float(np.max(y_arr[feasible_mask]))
        else:
            best_feasible = -np.inf

        y_mean = float(np.mean(y_arr))
        y_std = float(np.std(y_arr) + 1e-8)
        y_max = float(np.max(y_arr))

        n_feasible = int(np.sum(feasible_mask))
        feasible_ratio = float(n_feasible) / float(len(c_arr))

        # ENHANCED: Constraint statistics
        c_mean = float(np.mean(c_arr))
        c_std = float(np.std(c_arr) + 1e-8)
        c_min = float(np.min(c_arr))
        last_c = float(c_arr[-1])

        last_point = X_hist[-1]

        if feasible_mask.any():
            best_idx = int(np.argmax(y_arr * feasible_mask))
            best_point = X_hist[best_idx]
            best_c = float(c_arr[best_idx])
        else:
            # ENHANCED: Use least infeasible point
            least_infeas_idx = int(np.argmin(c_arr))
            best_point = X_hist[least_infeas_idx]
            best_c = float(c_arr[least_infeas_idx])

        if np.isneginf(best_feasible):
            best_feasible = 0.0

        progress = float(step) / float(budget)

        summary = np.array(
            [
                best_feasible,
                y_mean,
                y_std,
                y_max,
                float(n_feasible),
                feasible_ratio,
                float(len(X_hist)),
                float(step),
                float(budget),
                float(self.dim),
                c_mean,      # NEW
                c_std,       # NEW
                c_min,       # NEW
                last_c,      # NEW
                best_c,      # NEW
                progress,    # NEW
            ],
            dtype=np.float32,
        )

        surrogate_feats = self._local_surrogate_features(X_hist, f_hist, c_hist)

        state = np.concatenate(
            [summary, surrogate_feats, last_point.astype(np.float32), best_point.astype(np.float32)]
        ).astype(np.float32)

        # Safety
        if len(state) < self.state_dim:
            pad = np.zeros(self.state_dim - len(state), dtype=np.float32)
            state = np.concatenate([state, pad])
        else:
            state = state[: self.state_dim]

        return state

    def suggest(
        self,
        X_hist: np.ndarray,
        f_hist: np.ndarray,
        c_hist: np.ndarray,
        step: int,
        budget: int,
    ) -> np.ndarray:
        """Returns a point in [0, 1]^dim."""
        state = self._build_state(X_hist, f_hist, c_hist, step, budget)
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.policy(state_t)

        action = mean  # deterministic at test time

        # Map from R^d to [0,1]^d via tanh + rescale
        action = torch.tanh(action)
        action = (action + 1.0) / 2.0
        x_unit = action.clamp(0.0, 1.0).cpu().numpy()[0]

        return x_unit


# -----------------------------------------------------------------------------
# BASELINES (from benchmark_rl.py)
# -----------------------------------------------------------------------------

class RandomSearchOptimizer:
    def __init__(self, dim: int, rng: np.random.Generator = None):
        self.dim = dim
        self.rng = rng if rng is not None else np.random.default_rng()

    def suggest(
        self,
        X_hist: np.ndarray,
        f_hist: np.ndarray,
        c_hist: np.ndarray,
        step: int,
        budget: int,
    ) -> np.ndarray:
        return self.rng.random(self.dim)


class QLogEIOptimizer:
    """qLogEI baseline with constrained optimization."""

    def __init__(self, dim: int):
        self.dim = dim
        self.device = DEVICE
        self.dtype = DTYPE
        self.mc_samples = 128
        self.num_restarts = 5
        self.raw_samples = 128

    def _build_model(self, X_hist, f_hist, c_hist):
        train_x = torch.tensor(X_hist, dtype=self.dtype, device=self.device)
        train_obj = (-torch.tensor(f_hist, dtype=self.dtype, device=self.device)).unsqueeze(-1)
        train_con = torch.tensor(c_hist, dtype=self.dtype, device=self.device).unsqueeze(-1)

        NOISE_SE = 1e-2
        train_yvar_obj = torch.full_like(train_obj, NOISE_SE ** 2)
        train_yvar_con = torch.full_like(train_con, NOISE_SE ** 2)

        model_obj = SingleTaskGP(train_x, train_obj, train_yvar_obj)
        model_con = SingleTaskGP(train_x, train_con, train_yvar_con)
        model = ModelListGP(model_obj, model_con)
        mll = SumMarginalLogLikelihood(model.likelihood, model)

        fit_gpytorch_mll(mll)
        return model

    def suggest(
        self,
        X_hist: np.ndarray,
        f_hist: np.ndarray,
        c_hist: np.ndarray,
        step: int,
        budget: int,
    ) -> np.ndarray:
        model = self._build_model(X_hist, f_hist, c_hist)

        def obj_callable(Z: torch.Tensor, X: torch.Tensor = None):
            return Z[..., 0]

        def constraint_callable(Z: torch.Tensor):
            return Z[..., 1]

        objective = GenericMCObjective(obj_callable)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))

        train_obj = (-torch.tensor(f_hist, dtype=self.dtype, device=self.device)).unsqueeze(-1)
        train_con = torch.tensor(c_hist, dtype=self.dtype, device=self.device).unsqueeze(-1)

        feas_mask = (train_con <= 0.0)
        if feas_mask.any():
            best_f = (train_obj * feas_mask).max()
        else:
            best_f = train_obj.max()

        acq = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            constraints=[constraint_callable],
        )

        bounds = torch.stack(
            [
                torch.zeros(self.dim, dtype=self.dtype, device=self.device),
                torch.ones(self.dim, dtype=self.dtype, device=self.device),
            ]
        )

        try:
            candidates, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                options={"batch_limit": 5, "maxiter": 200},
            )
            x_unit = candidates.detach().cpu().numpy()[0]
        except Exception as e:
            print(f"[qLogEI] optimize_acqf failed with {e}, falling back to random.")
            x_unit = np.random.rand(self.dim)

        x_unit = np.clip(x_unit, 0.0, 1.0)
        return x_unit


# -----------------------------------------------------------------------------
# COCO EVALUATION UTILS
# -----------------------------------------------------------------------------

def map_unit_to_coco(x_unit: np.ndarray, problem) -> np.ndarray:
    lb = np.asarray(problem.lower_bounds, dtype=float)
    ub = np.asarray(problem.upper_bounds, dtype=float)
    return lb + x_unit * (ub - lb)


def eval_coco(problem, x_unit: np.ndarray) -> Tuple[float, float, np.ndarray]:
    x_real = map_unit_to_coco(x_unit, problem)
    f_val = problem(x_real)
    f = float(f_val)

    if hasattr(problem, "constraint"):
        g_vals = np.asarray(problem.constraint(x_real), dtype=float)
        if g_vals.size == 0:
            c_agg = 0.0
        else:
            c_agg = float(g_vals.max())
    else:
        c_agg = 0.0

    return f, c_agg, x_real


def run_single_algorithm(
    problem,
    dim: int,
    algo_name: str,
    optimizer_obj,
    budget: int,
    n_init: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    X_hist = []
    f_hist = []
    c_hist = []
    best_feasible = np.inf
    best_feasible_hist = []

    # Initial random design
    for _ in range(n_init):
        x_unit = rng.random(dim)
        f, c, _ = eval_coco(problem, x_unit)
        X_hist.append(x_unit)
        f_hist.append(f)
        c_hist.append(c)
        if c <= 0.0 and f < best_feasible:
            best_feasible = f
        best_feasible_hist.append(best_feasible)

    X_hist = np.asarray(X_hist, dtype=float)
    f_hist = np.asarray(f_hist, dtype=float)
    c_hist = np.asarray(c_hist, dtype=float)

    # Sequential loop
    for step in range(n_init, budget):
        x_unit = optimizer_obj.suggest(
            X_hist, f_hist, c_hist, step=step, budget=budget
        )
        f, c, _ = eval_coco(problem, x_unit)
        X_hist = np.vstack([X_hist, x_unit])
        f_hist = np.concatenate([f_hist, [f]])
        c_hist = np.concatenate([c_hist, [c]])

        if c <= 0.0 and f < best_feasible:
            best_feasible = f
        best_feasible_hist.append(best_feasible)

    return {
        "best_feasible": np.array(best_feasible_hist, dtype=float),
        "f_hist": f_hist,
        "c_hist": c_hist,
    }


# -----------------------------------------------------------------------------
# MAIN BENCHMARK LOOP
# -----------------------------------------------------------------------------

def run_benchmarks():
    suite = cocoex.Suite("bbob-constrained", "", "")
    results_rows = []
    summary_rows = []

    for dim in DIMENSIONS:
        budget = BUDGET_FACTOR * dim
        n_init = N_INIT_FACTOR * dim
        print(f"\n=== DIM = {dim}, budget = {budget}, n_init = {n_init} ===")

        rl_ckpt_path = MODEL_DIR / MODEL_PATTERN.format(dim=dim)
        if rl_ckpt_path.exists():
            rl_available = True
            rl_optimizer_template = FullCocoRLOptimizer(
                dim=dim,
                ckpt_path=rl_ckpt_path,
            )
            print(f"[INFO] Loaded FULL COCO RL model: {rl_ckpt_path}")
        else:
            rl_available = False
            print(f"[WARN] FULL COCO RL model not found for dim={dim}: {rl_ckpt_path}")

        for fid in FUNCTIONS:
            for inst in INSTANCES:
                print(f"\nProblem F{fid}, instance {inst}, dim {dim}")

                problem = suite.get_problem_by_function_dimension_instance(
                    fid, dim, inst
                )

                for rep in range(REPETITIONS):
                    print(f"  Repetition {rep+1}/{REPETITIONS}...")

                    seed = 1234 + rep

                    # --- Random Search ---
                    rs_rng = np.random.default_rng(seed=seed)
                    rs_opt = RandomSearchOptimizer(dim=dim, rng=rs_rng)
                    rs_hist = run_single_algorithm(
                        problem=problem,
                        dim=dim,
                        algo_name="Random",
                        optimizer_obj=rs_opt,
                        budget=budget,
                        n_init=n_init,
                        rng=rs_rng,
                    )

                    for t, bf in enumerate(rs_hist["best_feasible"]):
                        results_rows.append(
                            dict(
                                method="Random",
                                dim=dim,
                                function=fid,
                                instance=inst,
                                repetition=rep,
                                eval=t + 1,
                                best_feasible=bf,
                            )
                        )

                    # --- qLogEI ---
                    qlogei_opt = QLogEIOptimizer(dim=dim)
                    q_hist = run_single_algorithm(
                        problem=problem,
                        dim=dim,
                        algo_name="qLogEI",
                        optimizer_obj=qlogei_opt,
                        budget=budget,
                        n_init=n_init,
                        rng=np.random.default_rng(seed=seed),
                    )

                    for t, bf in enumerate(q_hist["best_feasible"]):
                        results_rows.append(
                            dict(
                                method="qLogEI",
                                dim=dim,
                                function=fid,
                                instance=inst,
                                repetition=rep,
                                eval=t + 1,
                                best_feasible=bf,
                            )
                        )

                    # --- Full COCO RL (if available) ---
                    if rl_available:
                        rl_opt = FullCocoRLOptimizer(
                            dim=dim,
                            ckpt_path=rl_ckpt_path,
                        )
                        rl_hist = run_single_algorithm(
                            problem=problem,
                            dim=dim,
                            algo_name="RL_Full",
                            optimizer_obj=rl_opt,
                            budget=budget,
                            n_init=n_init,
                            rng=np.random.default_rng(seed=seed),
                        )

                        for t, bf in enumerate(rl_hist["best_feasible"]):
                            results_rows.append(
                                dict(
                                    method="RL_Full",
                                    dim=dim,
                                    function=fid,
                                    instance=inst,
                                    repetition=rep,
                                    eval=t + 1,
                                    best_feasible=bf,
                                )
                            )

                    # Summary
                    rs_final = float(rs_hist["best_feasible"][-1])
                    q_final = float(q_hist["best_feasible"][-1])
                    rl_final = None

                    if rl_available:
                        rl_final = float(rl_hist["best_feasible"][-1])

                    if rl_available:
                        print(
                            f"    Summary (dim={dim}, F{fid}, inst={inst}, rep={rep+1}): "
                            f"Random={rs_final:.3e}, qLogEI={q_final:.3e}, RL_Full={rl_final:.3e}"
                        )
                    else:
                        print(
                            f"    Summary (dim={dim}, F{fid}, inst={inst}, rep={rep+1}): "
                            f"Random={rs_final:.3e}, qLogEI={q_final:.3e}, RL_Full=N/A"
                        )

                    summary_rows.append(
                        dict(
                            method="Random",
                            dim=dim,
                            function=fid,
                            instance=inst,
                            repetition=rep,
                            final_best_feasible=rs_final,
                        )
                    )
                    summary_rows.append(
                        dict(
                            method="qLogEI",
                            dim=dim,
                            function=fid,
                            instance=inst,
                            repetition=rep,
                            final_best_feasible=q_final,
                        )
                    )
                    if rl_available:
                        summary_rows.append(
                            dict(
                                method="RL_Full",
                                dim=dim,
                                function=fid,
                                instance=inst,
                                repetition=rep,
                                final_best_feasible=rl_final,
                            )
                        )

                problem.free()

    df = pd.DataFrame(results_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved benchmark results to: {OUT_CSV}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_CSV_SUMMARY, index=False)
    print(f"Saved per-repetition summary to: {OUT_CSV_SUMMARY}")


if __name__ == "__main__":
    run_benchmarks()
