"""
autoresearch.py
---------------
Evolutionary auto-tuning of PFC model parameters.

Inspired by Karpathy's autoresearch loop (applied to markets by Chris Worsey):
    - "Prompts are the weights, Sharpe is the loss function"
    - For PFC: "Agent parameters are the weights, RMSE vs spot is the loss"

Architecture:
    5 specialized agents, each controlling a model component:
        1. SeasonalAgent    — lookback window, seasonal smoothing
        2. WeekdayAgent     — f_W estimation parameters
        3. HourlyAgent      — f_H Gaussian sigma, lookback
        4. IntradayAgent    — f_Q Ridge alpha, correction strength
        5. LevelAgent       — forward proxy anchor, decay rate

    Weekly evolution loop:
        1. Backtest all agents with current parameters
        2. Identify worst-contributing agent (highest marginal RMSE)
        3. Perturb its parameters (single targeted modification)
        4. Re-run backtest with perturbed parameters
        5. If RMSE improved -> keep (commit), else -> revert
        6. Update Darwinian weights (good agents gain trust)

    Over time, parameters converge to values that minimize prediction
    error on realized spot prices. No LLM API required — pure
    algorithmic evolution with market data as feedback.

Usage:
    from pfc_shaping.pipeline.autoresearch import AutoResearchLoop
    loop = AutoResearchLoop.from_config(config)
    result = loop.evolve(epex_history, n_iterations=10)
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Agent definitions
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentParams:
    """Parameters controlled by a single agent."""
    values: dict[str, float]
    bounds: dict[str, tuple[float, float]]
    description: str = ""

    def perturb(self, rng: np.random.Generator) -> "AgentParams":
        """Create a perturbed copy (single random parameter change)."""
        new = copy.deepcopy(self)
        keys = list(new.values.keys())
        key = rng.choice(keys)
        lo, hi = new.bounds[key]
        current = new.values[key]

        # Perturbation: +-10-30% of range, clamped to bounds
        range_size = hi - lo
        delta = rng.uniform(-0.3, 0.3) * range_size
        new.values[key] = float(np.clip(current + delta, lo, hi))

        logger.info(
            "Perturbed %s: %.4f -> %.4f (bounds [%.4f, %.4f])",
            key, current, new.values[key], lo, hi,
        )
        return new


# Default agent configurations
SEASONAL_AGENT = AgentParams(
    values={
        "lookback_months": 36,
        "seasonal_smooth_sigma": 0.5,
    },
    bounds={
        "lookback_months": (12, 60),
        "seasonal_smooth_sigma": (0.1, 2.0),
    },
    description="Controls seasonal shape estimation lookback and smoothing",
)

WEEKDAY_AGENT = AgentParams(
    values={
        "fw_smooth_sigma_hours": 6.0,
        "holiday_weight": 0.75,
    },
    bounds={
        "fw_smooth_sigma_hours": (2.0, 12.0),
        "holiday_weight": (0.60, 0.90),
    },
    description="Controls weekday shape smoothing and holiday treatment",
)

HOURLY_AGENT = AgentParams(
    values={
        "gaussian_sigma": 0.5,
        "fh_lookback_months": 36,
    },
    bounds={
        "gaussian_sigma": (0.1, 2.0),
        "fh_lookback_months": (12, 60),
    },
    description="Controls hourly shape f_H Gaussian smoothing and lookback",
)

INTRADAY_AGENT = AgentParams(
    values={
        "ridge_alpha": 1.0,
        "correction_strength": 1.0,
    },
    bounds={
        "ridge_alpha": (0.01, 100.0),
        "correction_strength": (0.0, 2.0),
    },
    description="Controls intra-hour f_Q Ridge regression and Layer 2 correction",
)

LEVEL_AGENT = AgentParams(
    values={
        "anchor_months": 6,
        "annual_decay_clip_lo": -0.15,
        "annual_decay_clip_hi": -0.01,
    },
    bounds={
        "anchor_months": (3, 12),
        "annual_decay_clip_lo": (-0.25, -0.05),
        "annual_decay_clip_hi": (-0.05, 0.0),
    },
    description="Controls forward proxy base price derivation",
)


@dataclass
class Agent:
    """A single autoresearch agent controlling one PFC component."""
    name: str
    params: AgentParams
    weight: float = 1.0
    sharpe: float = 0.0
    rmse_contribution: float = 0.0
    n_improvements: int = 0
    n_reverts: int = 0

    # Darwinian weight bounds
    WEIGHT_MIN: float = 0.3
    WEIGHT_MAX: float = 2.5

    def update_weight(self, is_top_quartile: bool) -> None:
        """Darwinian weight update: reward good agents, penalize bad ones."""
        if is_top_quartile:
            self.weight = min(self.weight * 1.05, self.WEIGHT_MAX)
        else:
            self.weight = max(self.weight * 0.95, self.WEIGHT_MIN)


# ══════════════════════════════════════════════════════════════════════════════
# Backtest engine
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """Result of a single backtest run."""
    rmse: float
    mae: float
    bias: float
    n_points: int
    per_agent_rmse: dict[str, float] = field(default_factory=dict)
    details: str = ""


def _run_backtest(
    epex: pd.DataFrame,
    agents: dict[str, Agent],
    config: dict,
    test_months: int = 3,
) -> BacktestResult:
    """Run walk-forward backtest using agent parameters.

    Uses the last `test_months` of spot data as out-of-sample test set.
    Builds PFC from data prior to the test window, then compares to
    realized spot prices.

    Args:
        epex: Full EPEX spot history with 'price_eur_mwh' column.
        agents: Dict of agent name -> Agent with current parameters.
        config: Full config dict (for non-agent parameters).
        test_months: Number of months for out-of-sample test.

    Returns:
        BacktestResult with RMSE, MAE, bias, and per-agent contributions.
    """
    from pfc_shaping.model.shape_hourly import ShapeHourly
    from pfc_shaping.model.shape_intraday import ShapeIntraday
    from pfc_shaping.model.assembler import PFCAssembler
    from pfc_shaping.model.uncertainty import Uncertainty
    from pfc_shaping.data.forward_proxy import derive_base_prices

    # Split data: train on history, test on recent
    cutoff = epex.index.max() - pd.DateOffset(months=test_months)
    train = epex[epex.index < cutoff]
    test = epex[epex.index >= cutoff]

    if len(train) < 96 * 180 or len(test) < 96 * 14:
        return BacktestResult(
            rmse=999.0, mae=999.0, bias=999.0, n_points=0,
            details="Insufficient data for backtest",
        )

    # Extract agent parameters
    level_p = agents["level"].params.values
    hourly_p = agents["hourly"].params.values
    intraday_p = agents["intraday"].params.values

    try:
        from pfc_shaping.data.calendar_ch import enrich_15min_index

        # Fit models on training data
        lookback = int(hourly_p.get("fh_lookback_months", 36))
        sigma = hourly_p.get("gaussian_sigma", 0.5)

        sh = ShapeHourly(sigma=sigma)
        lb_cutoff = train.index.max() - pd.DateOffset(months=lookback)
        train_lb = train[train.index >= lb_cutoff]

        # Build calendar for training data
        cal_train = enrich_15min_index(train_lb.index)

        sh.fit(train_lb, cal_train)

        si = ShapeIntraday()
        si.fit(train_lb, entso_df=None, calendar_df=cal_train)

        unc = Uncertainty()
        unc.fit(train_lb, cal_train)

        # Derive base prices from training data
        anchor_m = int(level_p.get("anchor_months", 6))
        base_prices = derive_base_prices(
            train, start_year=cutoff.year, n_years=1,
            anchor_months=anchor_m,
        )

        # Build PFC over test period (no cascader/calibrator needed for backtest)
        assembler = PFCAssembler(
            shape_hourly=sh,
            shape_intraday=si,
            uncertainty=unc,
        )

        pfc = assembler.build(
            base_prices=base_prices,
            start_date=cutoff.strftime("%Y-%m-%d"),
            horizon_days=test_months * 31,
        )

        # Align PFC with test data
        common_idx = pfc.index.intersection(test.index)
        if len(common_idx) < 96 * 7:
            return BacktestResult(
                rmse=999.0, mae=999.0, bias=999.0, n_points=len(common_idx),
                details="Insufficient overlap between PFC and test data",
            )

        pfc_aligned = pfc.loc[common_idx, "price_shape"]
        spot_aligned = test.loc[common_idx, "price_eur_mwh"]

        errors = pfc_aligned.values - spot_aligned.values
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        bias = float(np.mean(errors))

        # Per-agent RMSE contribution (marginal analysis)
        # Decompose error by time-of-day (hourly agent), day-of-week (weekday),
        # season (seasonal), etc.
        idx_zh = common_idx.tz_convert("Europe/Zurich")
        per_agent = {}

        # Hourly agent: RMSE by hour
        hourly_rmse = []
        for h in range(24):
            mask = idx_zh.hour == h
            if mask.sum() > 0:
                hourly_rmse.append(np.sqrt(np.mean(errors[mask]**2)))
        per_agent["hourly"] = float(np.mean(hourly_rmse)) if hourly_rmse else rmse

        # Weekday agent: RMSE by day type
        weekday_rmse = []
        for dow in range(7):
            mask = idx_zh.dayofweek == dow
            if mask.sum() > 0:
                weekday_rmse.append(np.sqrt(np.mean(errors[mask]**2)))
        per_agent["weekday"] = float(np.std(weekday_rmse)) if weekday_rmse else rmse

        # Seasonal agent: RMSE by month
        monthly_bias = []
        for m in range(1, 13):
            mask = idx_zh.month == m
            if mask.sum() > 0:
                monthly_bias.append(abs(np.mean(errors[mask])))
        per_agent["seasonal"] = float(np.mean(monthly_bias)) if monthly_bias else rmse

        # Level agent: overall bias (systematic level error)
        per_agent["level"] = abs(bias)

        # Intraday agent: RMSE at 15min resolution vs hourly average
        pfc_hourly = pfc_aligned.resample("h").mean()
        per_agent["intraday"] = float(np.sqrt(np.mean(
            (pfc_aligned.values - np.repeat(pfc_hourly.values, 4)[:len(pfc_aligned)])**2
        ))) if len(pfc_hourly) > 0 else rmse

        return BacktestResult(
            rmse=rmse, mae=mae, bias=bias,
            n_points=len(common_idx),
            per_agent_rmse=per_agent,
            details=f"Test: {common_idx.min().date()} -> {common_idx.max().date()}",
        )

    except Exception as exc:
        logger.error("Backtest failed: %s", exc)
        return BacktestResult(
            rmse=999.0, mae=999.0, bias=999.0, n_points=0,
            details=f"Error: {exc}",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Evolution loop
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvolutionResult:
    """Result of the full evolution loop."""
    iterations: int
    improvements: int
    reverts: int
    initial_rmse: float
    final_rmse: float
    improvement_pct: float
    agent_history: list[dict]
    best_params: dict[str, dict]


class AutoResearchLoop:
    """Evolutionary auto-tuning of PFC model parameters.

    Implements Karpathy-style autoresearch adapted for energy forward curves:
    agents control model parameters, RMSE vs spot is the loss function,
    and a Darwinian selection process keeps improving modifications.
    """

    def __init__(
        self,
        agents: dict[str, Agent] | None = None,
        config: dict | None = None,
        seed: int = 42,
        state_path: str | None = None,
    ) -> None:
        self.agents = agents or self._default_agents()
        self.config = config or {}
        self.rng = np.random.default_rng(seed)
        self.state_path = Path(state_path) if state_path else None
        self.history: list[dict] = []

    @classmethod
    def from_config(cls, config: dict) -> "AutoResearchLoop":
        """Create loop from config.yaml."""
        ar_cfg = config.get("autoresearch", {})
        default_state = str(Path(__file__).resolve().parent.parent / "model" / "artifacts" / "autoresearch_state.json")
        state_path = ar_cfg.get("state_path", default_state)
        loop = cls(config=config, state_path=state_path)

        # Load saved state if exists
        if loop.state_path and loop.state_path.exists():
            loop.load_state()
            logger.info("Loaded autoresearch state: %d prior iterations", len(loop.history))

        return loop

    @staticmethod
    def _default_agents() -> dict[str, Agent]:
        return {
            "seasonal": Agent(name="seasonal", params=copy.deepcopy(SEASONAL_AGENT)),
            "weekday": Agent(name="weekday", params=copy.deepcopy(WEEKDAY_AGENT)),
            "hourly": Agent(name="hourly", params=copy.deepcopy(HOURLY_AGENT)),
            "intraday": Agent(name="intraday", params=copy.deepcopy(INTRADAY_AGENT)),
            "level": Agent(name="level", params=copy.deepcopy(LEVEL_AGENT)),
        }

    def evolve(
        self,
        epex: pd.DataFrame,
        n_iterations: int = 10,
        test_months: int = 3,
    ) -> EvolutionResult:
        """Run the evolution loop.

        Args:
            epex: Full EPEX spot history.
            n_iterations: Number of evolution steps.
            test_months: Months of out-of-sample test data.

        Returns:
            EvolutionResult with full history and best parameters.
        """
        logger.info("=" * 70)
        logger.info("AUTORESEARCH: Starting evolution loop (%d iterations)", n_iterations)
        logger.info("=" * 70)

        # Initial backtest
        baseline = _run_backtest(epex, self.agents, self.config, test_months)
        initial_rmse = baseline.rmse
        current_rmse = baseline.rmse

        logger.info(
            "Baseline: RMSE=%.2f, MAE=%.2f, Bias=%.2f (%d points)",
            baseline.rmse, baseline.mae, baseline.bias, baseline.n_points,
        )

        if baseline.rmse >= 999:
            logger.error("Baseline backtest failed — aborting evolution")
            return EvolutionResult(
                iterations=0, improvements=0, reverts=0,
                initial_rmse=999, final_rmse=999, improvement_pct=0,
                agent_history=[], best_params={},
            )

        improvements = 0
        reverts = 0

        for iteration in range(1, n_iterations + 1):
            logger.info("--- Iteration %d/%d ---", iteration, n_iterations)

            # 1. Select agent to perturb (weighted by contribution + exploration)
            #    80% exploit (weight by per-agent RMSE), 20% explore (random)
            if baseline.per_agent_rmse and self.rng.random() > 0.2:
                # Weight selection by contribution (higher = more likely chosen)
                names = list(baseline.per_agent_rmse.keys())
                scores = np.array([baseline.per_agent_rmse[n] for n in names])
                # Softmax-like selection to avoid always picking the same agent
                scores = scores / (scores.sum() + 1e-8)
                worst_name = self.rng.choice(names, p=scores)
            else:
                worst_name = self.rng.choice(list(self.agents.keys()))

            worst_agent = self.agents[worst_name]
            logger.info(
                "Worst agent: %s (contribution=%.3f, weight=%.2f)",
                worst_name,
                baseline.per_agent_rmse.get(worst_name, 0),
                worst_agent.weight,
            )

            # 2. Perturb worst agent's parameters
            old_params = copy.deepcopy(worst_agent.params)
            new_params = old_params.perturb(self.rng)
            worst_agent.params = new_params

            # 3. Re-run backtest
            trial = _run_backtest(epex, self.agents, self.config, test_months)

            # 4. Keep or revert
            if trial.rmse < current_rmse:
                delta = current_rmse - trial.rmse
                current_rmse = trial.rmse
                baseline = trial
                improvements += 1
                worst_agent.n_improvements += 1
                worst_agent.rmse_contribution = trial.per_agent_rmse.get(worst_name, 0)

                logger.info(
                    "KEEP: RMSE %.2f -> %.2f (delta=%.3f) [%s]",
                    current_rmse + delta, current_rmse, delta, worst_name,
                )
            else:
                worst_agent.params = old_params
                reverts += 1
                worst_agent.n_reverts += 1

                logger.info(
                    "REVERT: trial RMSE %.2f >= current %.2f [%s]",
                    trial.rmse, current_rmse, worst_name,
                )

            # 5. Darwinian weight update
            agent_scores = {
                name: agent.rmse_contribution
                for name, agent in self.agents.items()
            }
            if agent_scores:
                sorted_agents = sorted(agent_scores, key=agent_scores.get)
                n = len(sorted_agents)
                top_q = set(sorted_agents[:max(1, n // 4)])

                for name, agent in self.agents.items():
                    agent.update_weight(name in top_q)

            # Record history
            self.history.append({
                "iteration": iteration,
                "target_agent": worst_name,
                "action": "keep" if trial.rmse < current_rmse + 0.001 else "revert",
                "rmse_before": float(current_rmse if trial.rmse < current_rmse + 0.001 else current_rmse),
                "rmse_after": float(trial.rmse),
                "agent_weights": {n: a.weight for n, a in self.agents.items()},
                "agent_params": {n: a.params.values for n, a in self.agents.items()},
            })

        # Final summary
        improvement_pct = (initial_rmse - current_rmse) / initial_rmse * 100 if initial_rmse > 0 else 0

        logger.info("=" * 70)
        logger.info("AUTORESEARCH COMPLETE")
        logger.info(
            "  RMSE: %.2f -> %.2f (%.1f%% improvement)",
            initial_rmse, current_rmse, improvement_pct,
        )
        logger.info("  Improvements: %d, Reverts: %d", improvements, reverts)
        for name, agent in self.agents.items():
            logger.info(
                "  Agent %s: weight=%.2f, kept=%d, reverted=%d, params=%s",
                name, agent.weight, agent.n_improvements, agent.n_reverts,
                agent.params.values,
            )
        logger.info("=" * 70)

        # Save state
        if self.state_path:
            self.save_state()

        return EvolutionResult(
            iterations=n_iterations,
            improvements=improvements,
            reverts=reverts,
            initial_rmse=initial_rmse,
            final_rmse=current_rmse,
            improvement_pct=improvement_pct,
            agent_history=self.history,
            best_params={n: a.params.values for n, a in self.agents.items()},
        )

    def save_state(self) -> None:
        """Persist agent state to JSON."""
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "agents": {
                name: {
                    "params": agent.params.values,
                    "weight": agent.weight,
                    "n_improvements": agent.n_improvements,
                    "n_reverts": agent.n_reverts,
                }
                for name, agent in self.agents.items()
            },
            "history": self.history[-100:],  # Keep last 100 iterations
        }
        self.state_path.write_text(json.dumps(state, indent=2, default=str))
        logger.info("Autoresearch state saved: %s", self.state_path)

    def load_state(self) -> None:
        """Load agent state from JSON."""
        if not self.state_path or not self.state_path.exists():
            return
        state = json.loads(self.state_path.read_text())
        for name, agent_state in state.get("agents", {}).items():
            if name in self.agents:
                self.agents[name].params.values.update(agent_state["params"])
                self.agents[name].weight = agent_state["weight"]
                self.agents[name].n_improvements = agent_state.get("n_improvements", 0)
                self.agents[name].n_reverts = agent_state.get("n_reverts", 0)
        self.history = state.get("history", [])

    def get_optimized_config(self) -> dict:
        """Extract optimized parameters as a config-compatible dict."""
        return {
            "model": {
                "lookback_months": int(self.agents["seasonal"].params.values["lookback_months"]),
                "gaussian_sigma": self.agents["hourly"].params.values["gaussian_sigma"],
            },
            "forward_proxy": {
                "anchor_months": int(self.agents["level"].params.values["anchor_months"]),
            },
            "autoresearch": {
                "agent_weights": {n: a.weight for n, a in self.agents.items()},
                "total_iterations": len(self.history),
                "total_improvements": sum(a.n_improvements for a in self.agents.values()),
                "total_reverts": sum(a.n_reverts for a in self.agents.values()),
            },
        }
