#!/usr/bin/env python3
"""
Experiment 2: PPO under IVT PAT observability constraints.

Experiment 2 changes:
- Retains the v8 10-action space with a dedicated masked stop action.
- Softens Mg feed costs so PPO is more willing to use Mg when it is limiting.
- Adds a terminal unused-capacity penalty for voluntary stopping while enzyme/substrate capacity remains.
- This discourages the previous conservative local optimum: feed NTP, avoid Mg, and stop at min_stop_step.

Dependencies:
    pip install torch numpy matplotlib

Smoke test:
    python experiment2_ivt_pat_observability.py --mode smoke

MacBook Air:
    python experiment2_ivt_pat_observability.py --mode all --total-steps 1500000 --num-envs 128 --device cpu

GTX 1080 Ti:
    python experiment2_ivt_pat_observability.py --mode all --total-steps 3000000 --num-envs 512 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


@dataclass
class EnvConfig:
    horizon: int = 64
    min_stop_step: int = 48
    dt: float = 1.0

    # Experiment 2 observation model.
    # Dynamics/reward/action space are held fixed from v8.
    obs_mode: str = "oracle"
    assay_interval: int = 8
    assay_noise: float = 0.03
    missing_prob: float = 0.10

    # Feed amounts in normalized arbitrary units.
    ntp_low_feed: float = 0.050
    ntp_high_feed: float = 0.120
    mg_low_feed: float = 0.040
    mg_high_feed: float = 0.100

    # State clamps.
    max_ntp: float = 2.50
    max_mg: float = 2.00
    max_ppi: float = 5.00
    max_mg_ppi: float = 5.00
    max_rna: float = 5.00

    # Nominal IVT kinetics.
    k_cat_nominal: float = 0.085
    ntp_consumption: float = 0.55

    # v2 change: Mg2+ is more likely to become limiting.
    mg_consumption: float = 0.45

    ppi_generation: float = 0.45

    # Precipitation / inhibition.
    precip_threshold: float = 0.40
    ppi_safe: float = 1.20
    mg_ppi_safe: float = 0.80
    mg_high_safe: float = 1.20

    # v8 change: discourage solving the task by flooding NTP to the cap.
    ntp_high_safe: float = 1.75
    penalty_ntp_high: float = 0.005

    k_mgppi_inhib: float = 0.50

    # pH model.
    ph_optimum: float = 7.50
    ph_safe_deviation: float = 0.35
    ph_drift_per_rate: float = 0.020
    ph_drift_per_ppi: float = 0.001

    # Product degradation.
    degradation_base: float = 0.0005

    # v2 change: stronger late/stress-driven degradation so endpoint control matters.
    degradation_stress_scale: float = 0.025
    age_degradation_start_frac: float = 0.70
    age_degradation_scale: float = 0.020

    # Dense reward. These discourage trivial overfeeding.
    alpha_yield: float = 1.0
    cost_ntp: float = 0.050
    cost_mg: float = 0.025
    cost_time: float = 0.0005
    penalty_stress: float = 0.035

    # Terminal reward.
    terminal_scale: float = 6.0
    terminal_ntp_cost: float = 0.05
    terminal_mg_cost: float = 0.035

    # v8 change: penalize unused terminal NTP to improve reagent efficiency.
    terminal_unused_ntp_cost: float = 0.02

    # v6/v8: penalize voluntarily stopping while productive capacity remains.
    # This makes endpoint control state-dependent rather than rewarding the first legal stop.
    stop_unused_capacity_penalty: float = 1.25

    lambda_ppi_quality: float = 0.15
    lambda_mgppi_quality: float = 0.60
    lambda_ph_quality: float = 0.20


@dataclass
class PPOConfig:
    total_steps: int = 1_000_000
    num_envs: int = 512
    rollout_steps: int = 64
    ppo_epochs: int = 4
    minibatches: int = 8
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.20
    entropy_coef: float = 0.010
    value_coef: float = 0.50
    max_grad_norm: float = 0.50
    hidden_size: int = 64


class IVTVectorEnv:
    """
    Vectorized lightweight IVT control environment.

    Hidden state:
        0 ntp
        1 mg_free
        2 ppi
        3 mg_ppi
        4 rna_full
        5 rna_trunc
        6 enzyme
        7 ph

    Observation:
        oracle normalized state + normalized time.

    Action:
        discrete action representing:
            actions 0-8: feed-only actions with
                ntp_feed in {none, low, high}
                mg_feed in {none, low, high}
            action 9: dedicated stop action

        total actions = 10
    """

    NTP = 0
    MG = 1
    PPI = 2
    MG_PPI = 3
    RNA_FULL = 4
    RNA_TRUNC = 5
    ENZYME = 6
    PH = 7

    def __init__(
        self,
        num_envs: int,
        cfg: EnvConfig,
        device: torch.device,
        seed: int = 0,
    ) -> None:
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device
        self.obs_dim = self._obs_dim_for_mode(cfg.obs_mode)
        self.n_actions = 10

        torch.manual_seed(seed)

        self.state = torch.zeros((num_envs, 8), device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

        self.total_ntp_fed = torch.zeros(num_envs, device=device)
        self.total_mg_fed = torch.zeros(num_envs, device=device)
        self.total_reward = torch.zeros(num_envs, device=device)

        # PAT/feed-history buffers used by non-oracle observation modes.
        self.prev_ntp_feed = torch.zeros(num_envs, device=device)
        self.prev_mg_feed = torch.zeros(num_envs, device=device)
        self.last_assay_ntp = torch.zeros(num_envs, device=device)
        self.last_assay_mg = torch.zeros(num_envs, device=device)
        self.last_assay_rna = torch.zeros(num_envs, device=device)
        self.last_assay_ppi = torch.zeros(num_envs, device=device)
        self.assay_age = torch.zeros(num_envs, device=device)
        self.assay_mask = torch.ones(num_envs, device=device)

        self.params: Dict[str, torch.Tensor] = {}

        self.feed_action_table = self._make_feed_action_table().to(device)
        self.stop_action = 9

        self.ntp_feed_amounts = torch.tensor(
            [0.0, cfg.ntp_low_feed, cfg.ntp_high_feed],
            dtype=torch.float32,
            device=device,
        )
        self.mg_feed_amounts = torch.tensor(
            [0.0, cfg.mg_low_feed, cfg.mg_high_feed],
            dtype=torch.float32,
            device=device,
        )

        self.reset()

    @staticmethod
    def _obs_dim_for_mode(obs_mode: str) -> int:
        if obs_mode == "oracle":
            return 9
        if obs_mode == "cheap_pat":
            return 7
        if obs_mode in {"delayed_atline", "noisy_missing"}:
            return 13
        raise ValueError(
            f"Unknown obs_mode={obs_mode!r}. Expected one of: "
            "oracle, cheap_pat, delayed_atline, noisy_missing."
        )

    @staticmethod
    def _make_feed_action_table() -> torch.Tensor:
        rows = []
        for ntp_level in range(3):
            for mg_level in range(3):
                rows.append([ntp_level, mg_level])
        return torch.tensor(rows, dtype=torch.long)

    @staticmethod
    def feed_action_index(ntp_level: int, mg_level: int) -> int:
        return ntp_level * 3 + mg_level

    @staticmethod
    def action_index(ntp_level: int, mg_level: int, stop: int = 0) -> int:
        # Backwards-compatible helper used by the baselines.
        # In v8, stop is a dedicated action and no longer carries feed levels.
        if stop:
            return 9
        return ntp_level * 3 + mg_level

    @staticmethod
    def decode_action_index(action_idx: int) -> Tuple[int, int, int]:
        if int(action_idx) == 9:
            return 0, 0, 1
        ntp = int(action_idx) // 3
        mg = int(action_idx) % 3
        return ntp, mg, 0

    def mask_action_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # Stop should not be sampled before it can actually terminate the episode.
        # Because step() increments step_count before checking done, action 9 can
        # legally stop when current step_count + 1 >= min_stop_step.
        can_select_stop = (self.step_count + 1) >= self.cfg.min_stop_step
        return mask_stop_logits_from_pre_step(
            logits=logits,
            pre_step_count=self.step_count,
            horizon=self.cfg.horizon,
            min_stop_step=self.cfg.min_stop_step,
        )

    def _ensure_params_exist(self) -> None:
        names = [
            "k_cat",
            "k_deact",
            "km_ntp",
            "km_mg",
            "k_ppi_inhib",
            "p_trunc_base",
            "p_trunc_ppi",
            "precip_rate",
        ]
        for name in names:
            if name not in self.params:
                self.params[name] = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        n = env_ids.numel()
        cfg = self.cfg
        dev = self.device

        self._ensure_params_exist()

        # v2 change: wider initial-condition variability so a single static schedule is less dominant.
        self.state[env_ids, self.NTP] = uniform(dev, n, 0.50, 1.30)
        self.state[env_ids, self.MG] = uniform(dev, n, 0.40, 1.10)
        self.state[env_ids, self.PPI] = uniform(dev, n, 0.00, 0.05)
        self.state[env_ids, self.MG_PPI] = 0.0
        self.state[env_ids, self.RNA_FULL] = 0.0
        self.state[env_ids, self.RNA_TRUNC] = 0.0
        self.state[env_ids, self.ENZYME] = uniform(dev, n, 0.60, 1.30)
        self.state[env_ids, self.PH] = uniform(dev, n, 7.30, 7.70)

        self.step_count[env_ids] = 0
        self.total_ntp_fed[env_ids] = 0.0
        self.total_mg_fed[env_ids] = 0.0
        self.total_reward[env_ids] = 0.0

        self.prev_ntp_feed[env_ids] = 0.0
        self.prev_mg_feed[env_ids] = 0.0
        self.assay_age[env_ids] = 0.0
        self.assay_mask[env_ids] = 1.0

        # v2 change: wider kinetic/process variability to reward adaptive control.
        self.params["k_cat"][env_ids] = cfg.k_cat_nominal * uniform(dev, n, 0.60, 1.40)
        self.params["k_deact"][env_ids] = uniform(dev, n, 0.004, 0.030)
        self.params["km_ntp"][env_ids] = uniform(dev, n, 0.10, 0.35)
        self.params["km_mg"][env_ids] = uniform(dev, n, 0.05, 0.25)
        self.params["k_ppi_inhib"][env_ids] = uniform(dev, n, 0.30, 3.00)
        self.params["p_trunc_base"][env_ids] = uniform(dev, n, 0.03, 0.14)
        self.params["p_trunc_ppi"][env_ids] = uniform(dev, n, 0.03, 0.12)
        self.params["precip_rate"][env_ids] = uniform(dev, n, 0.01, 0.10)

        self._write_assay_values(env_ids, noisy=(cfg.obs_mode == "noisy_missing"))

        return self.get_obs()

    def get_obs(self) -> torch.Tensor:
        cfg = self.cfg
        s = self.state
        time_norm = self.step_count.float() / cfg.horizon

        if cfg.obs_mode == "oracle":
            obs = torch.stack(
                [
                    s[:, self.NTP] / cfg.max_ntp,
                    s[:, self.MG] / cfg.max_mg,
                    s[:, self.PPI] / cfg.max_ppi,
                    s[:, self.MG_PPI] / cfg.max_mg_ppi,
                    s[:, self.RNA_FULL] / cfg.max_rna,
                    s[:, self.RNA_TRUNC] / cfg.max_rna,
                    s[:, self.ENZYME] / 1.30,
                    (s[:, self.PH] - cfg.ph_optimum) / 1.0,
                    time_norm,
                ],
                dim=-1,
            )
            return obs.clamp(-5.0, 5.0)

        cheap_core = self._cheap_pat_core()

        if cfg.obs_mode == "cheap_pat":
            obs = torch.cat([cheap_core, time_norm[:, None]], dim=-1)
            return obs.clamp(-5.0, 5.0)

        if cfg.obs_mode in {"delayed_atline", "noisy_missing"}:
            assay_obs = torch.stack(
                [
                    self.last_assay_ntp / cfg.max_ntp,
                    self.last_assay_mg / cfg.max_mg,
                    self.last_assay_rna / cfg.max_rna,
                    self.last_assay_ppi / cfg.max_ppi,
                    self.assay_age / cfg.horizon,
                    self.assay_mask,
                ],
                dim=-1,
            )
            obs = torch.cat([cheap_core, assay_obs, time_norm[:, None]], dim=-1)
            return obs.clamp(-5.0, 5.0)

        raise ValueError(f"Unknown obs_mode={cfg.obs_mode!r}")

    def _cheap_pat_core(self) -> torch.Tensor:
        cfg = self.cfg
        s = self.state

        conductivity = (
            0.45 * s[:, self.NTP]
            + 0.35 * s[:, self.MG]
            + 0.25 * s[:, self.PPI]
            - 0.15 * s[:, self.MG_PPI]
        )

        total_ntp_scale = cfg.horizon * cfg.ntp_high_feed + 1e-8
        total_mg_scale = cfg.horizon * cfg.mg_high_feed + 1e-8

        return torch.stack(
            [
                (s[:, self.PH] - cfg.ph_optimum) / 1.0,
                conductivity / 2.0,
                self.prev_ntp_feed / (cfg.ntp_high_feed + 1e-8),
                self.prev_mg_feed / (cfg.mg_high_feed + 1e-8),
                self.total_ntp_fed / total_ntp_scale,
                self.total_mg_fed / total_mg_scale,
            ],
            dim=-1,
        )

    @torch.no_grad()
    def _write_assay_values(self, env_ids: torch.Tensor, noisy: bool) -> None:
        cfg = self.cfg
        s = self.state
        ids = env_ids

        ntp = s[ids, self.NTP]
        mg = s[ids, self.MG]
        rna = s[ids, self.RNA_FULL] + s[ids, self.RNA_TRUNC]
        ppi = s[ids, self.PPI]

        if noisy:
            ntp = add_relative_noise(ntp, cfg.assay_noise).clamp(0.0, cfg.max_ntp)
            mg = add_relative_noise(mg, cfg.assay_noise).clamp(0.0, cfg.max_mg)
            rna = add_relative_noise(rna, cfg.assay_noise).clamp(0.0, cfg.max_rna)
            ppi = add_relative_noise(ppi, cfg.assay_noise).clamp(0.0, cfg.max_ppi)

        self.last_assay_ntp[ids] = ntp
        self.last_assay_mg[ids] = mg
        self.last_assay_rna[ids] = rna
        self.last_assay_ppi[ids] = ppi
        self.assay_age[ids] = 0.0
        self.assay_mask[ids] = 1.0

    @torch.no_grad()
    def _update_pat_after_step(self, ntp_feed: torch.Tensor, mg_feed: torch.Tensor) -> None:
        cfg = self.cfg

        self.prev_ntp_feed = ntp_feed.clone()
        self.prev_mg_feed = mg_feed.clone()

        if cfg.obs_mode not in {"delayed_atline", "noisy_missing"}:
            return

        self.assay_age += 1.0
        self.assay_mask.zero_()

        scheduled = (self.step_count % cfg.assay_interval) == 0

        if cfg.obs_mode == "noisy_missing":
            available = scheduled & (torch.rand(self.num_envs, device=self.device) > cfg.missing_prob)
            noisy = True
        else:
            available = scheduled
            noisy = False

        if available.any():
            ids = torch.where(available)[0]
            self._write_assay_values(ids, noisy=noisy)

    @torch.no_grad()
    def step(
        self,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        dt = cfg.dt
        eps = 1e-8

        actions = actions.long()
        stop_signal = actions == self.stop_action

        feed_actions = torch.clamp(actions, min=0, max=8)
        action_components = self.feed_action_table[feed_actions]
        ntp_levels = action_components[:, 0]
        mg_levels = action_components[:, 1]

        # Dedicated stop action carries no feed. If a random or external policy
        # chooses stop before min_stop_step, the environment simply continues
        # with no feed; PPO itself masks this action during training/evaluation.
        ntp_levels = torch.where(stop_signal, torch.zeros_like(ntp_levels), ntp_levels)
        mg_levels = torch.where(stop_signal, torch.zeros_like(mg_levels), mg_levels)

        ntp_feed = self.ntp_feed_amounts[ntp_levels]
        mg_feed = self.mg_feed_amounts[mg_levels]

        s = self.state
        old_full = s[:, self.RNA_FULL].clone()

        # Apply feeds before reaction update.
        s[:, self.NTP] = (s[:, self.NTP] + ntp_feed).clamp(0.0, cfg.max_ntp)
        s[:, self.MG] = (s[:, self.MG] + mg_feed).clamp(0.0, cfg.max_mg)

        self.total_ntp_fed += ntp_feed
        self.total_mg_fed += mg_feed

        ntp = s[:, self.NTP]
        mg = s[:, self.MG]
        ppi = s[:, self.PPI]
        mg_ppi = s[:, self.MG_PPI]
        full = s[:, self.RNA_FULL]
        trunc = s[:, self.RNA_TRUNC]
        enzyme = s[:, self.ENZYME]
        ph = s[:, self.PH]

        f_ntp = michaelis_menten(ntp, self.params["km_ntp"])
        f_mg = michaelis_menten(mg, self.params["km_mg"])
        f_ppi = 1.0 / (1.0 + self.params["k_ppi_inhib"] * ppi)
        f_ph = torch.exp(-2.0 * torch.abs(ph - cfg.ph_optimum))
        f_mgppi = 1.0 / (1.0 + cfg.k_mgppi_inhib * mg_ppi)

        r_raw = self.params["k_cat"] * enzyme * f_ntp * f_mg * f_ppi * f_ph * f_mgppi

        # Prevent impossible consumption.
        max_r_by_ntp = 0.95 * ntp / (cfg.ntp_consumption * dt + eps)
        max_r_by_mg = 0.95 * mg / (cfg.mg_consumption * dt + eps)
        r_ivt = torch.minimum(r_raw, torch.minimum(max_r_by_ntp, max_r_by_mg))

        # Truncation risk increases with PPi and low NTP.
        low_ntp_stress = torch.relu(0.20 - ntp)
        ppi_trunc_signal = ppi / (1.0 + ppi)

        p_trunc = (
            self.params["p_trunc_base"]
            + self.params["p_trunc_ppi"] * ppi_trunc_signal
            + 0.05 * low_ntp_stress
        ).clamp(0.01, 0.70)

        delta_full = dt * r_ivt * (1.0 - p_trunc)
        delta_trunc = dt * r_ivt * p_trunc

        new_ntp = ntp - dt * cfg.ntp_consumption * r_ivt
        new_mg = mg - dt * cfg.mg_consumption * r_ivt
        new_ppi = ppi + dt * cfg.ppi_generation * r_ivt

        # Mg-PPi precipitation.
        precip_drive = torch.relu(new_mg * new_ppi - cfg.precip_threshold)
        precip_flux = dt * self.params["precip_rate"] * precip_drive
        precip_flux = torch.minimum(precip_flux, 0.95 * torch.minimum(new_mg, new_ppi))

        new_mg = new_mg - precip_flux
        new_ppi = new_ppi - precip_flux
        new_mg_ppi = mg_ppi + precip_flux

        # pH drift from reaction progress and PPi burden.
        new_ph = ph - dt * cfg.ph_drift_per_rate * r_ivt - dt * cfg.ph_drift_per_ppi * new_ppi

        # Product degradation under stress and late reaction age.
        ph_stress = torch.relu(torch.abs(new_ph - cfg.ph_optimum) - cfg.ph_safe_deviation)
        ppi_stress = torch.relu(new_ppi - cfg.ppi_safe)
        mgppi_stress = torch.relu(new_mg_ppi - cfg.mg_ppi_safe)

        age_frac = self.step_count.float() / cfg.horizon
        age_stress = torch.relu(age_frac - cfg.age_degradation_start_frac)

        degradation_rate = (
            cfg.degradation_base
            + cfg.degradation_stress_scale * (ph_stress + ppi_stress + mgppi_stress)
            + cfg.age_degradation_scale * age_stress
        )
        degradation_factor = torch.exp(-degradation_rate * dt)

        new_full = (full + delta_full) * degradation_factor
        new_trunc = (trunc + delta_trunc) * degradation_factor

        # Enzyme deactivation remains first-order, but wider reset variability makes it matter more.
        new_enzyme = enzyme * torch.exp(-self.params["k_deact"] * dt)

        s[:, self.NTP] = new_ntp.clamp(0.0, cfg.max_ntp)
        s[:, self.MG] = new_mg.clamp(0.0, cfg.max_mg)
        s[:, self.PPI] = new_ppi.clamp(0.0, cfg.max_ppi)
        s[:, self.MG_PPI] = new_mg_ppi.clamp(0.0, cfg.max_mg_ppi)
        s[:, self.RNA_FULL] = new_full.clamp(0.0, cfg.max_rna)
        s[:, self.RNA_TRUNC] = new_trunc.clamp(0.0, cfg.max_rna)
        s[:, self.ENZYME] = new_enzyme.clamp(0.0, 2.0)
        s[:, self.PH] = new_ph.clamp(6.0, 9.0)

        self.step_count += 1
        self._update_pat_after_step(ntp_feed=ntp_feed, mg_feed=mg_feed)

        net_delta_full = s[:, self.RNA_FULL] - old_full

        ntp_high_stress = torch.relu(s[:, self.NTP] - cfg.ntp_high_safe)

        stress = (
            torch.relu(s[:, self.PPI] - cfg.ppi_safe)
            + torch.relu(s[:, self.MG_PPI] - cfg.mg_ppi_safe)
            + torch.relu(s[:, self.MG] - cfg.mg_high_safe)
            + torch.relu(torch.abs(s[:, self.PH] - cfg.ph_optimum) - cfg.ph_safe_deviation)
        )

        reward = (
            cfg.alpha_yield * net_delta_full
            - cfg.cost_ntp * ntp_feed
            - cfg.cost_mg * mg_feed
            - cfg.cost_time
            - cfg.penalty_stress * stress
            - cfg.penalty_ntp_high * ntp_high_stress
        )

        # v8: stop is a dedicated action. The environment still guards against
        # early external stop actions, while PPO masks them before sampling.
        can_stop = self.step_count >= cfg.min_stop_step
        effective_stop = stop_signal & can_stop
        done = effective_stop | (self.step_count >= cfg.horizon)

        metrics = self.compute_metrics()

        unused_capacity = (
            s[:, self.ENZYME]
            * torch.minimum(s[:, self.NTP], s[:, self.MG])
        )

        voluntary_early_stop = effective_stop & (self.step_count < cfg.horizon)

        terminal_reward = (
            cfg.terminal_scale * metrics["qa_yield"]
            - cfg.terminal_ntp_cost * self.total_ntp_fed
            - cfg.terminal_mg_cost * self.total_mg_fed
            - cfg.terminal_unused_ntp_cost * s[:, self.NTP]
            - voluntary_early_stop.float()
            * cfg.stop_unused_capacity_penalty
            * unused_capacity
        )

        reward = reward + done.float() * terminal_reward
        self.total_reward += reward

        info = {
            **metrics,
            "total_reward": self.total_reward.clone(),
            "total_ntp_fed": self.total_ntp_fed.clone(),
            "total_mg_fed": self.total_mg_fed.clone(),
            "stop_time": self.step_count.float().clone(),
        }

        return self.get_obs(), reward, done, info

    @torch.no_grad()
    def compute_metrics(self) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        s = self.state
        eps = 1e-8

        full = s[:, self.RNA_FULL]
        trunc = s[:, self.RNA_TRUNC]
        ppi = s[:, self.PPI]
        mg_ppi = s[:, self.MG_PPI]
        ph = s[:, self.PH]

        integrity = full / (full + trunc + eps)

        impurity_quality = torch.exp(
            -cfg.lambda_ppi_quality * ppi
            -cfg.lambda_mgppi_quality * mg_ppi
            -cfg.lambda_ph_quality * torch.abs(ph - cfg.ph_optimum)
        )

        qa_yield = full * integrity * impurity_quality

        return {
            "qa_yield": qa_yield.clone(),
            "rna_full": full.clone(),
            "rna_trunc": trunc.clone(),
            "integrity": integrity.clone(),
            "ppi": ppi.clone(),
            "mg_ppi": mg_ppi.clone(),
            "ph": ph.clone(),
            "enzyme": s[:, self.ENZYME].clone(),
        }

    @torch.no_grad()
    def raw_state_dict(self, env_id: int = 0) -> Dict[str, float]:
        s = self.state[env_id]
        return {
            "ntp": float(s[self.NTP].detach().cpu()),
            "mg_free": float(s[self.MG].detach().cpu()),
            "ppi": float(s[self.PPI].detach().cpu()),
            "mg_ppi": float(s[self.MG_PPI].detach().cpu()),
            "rna_full": float(s[self.RNA_FULL].detach().cpu()),
            "rna_trunc": float(s[self.RNA_TRUNC].detach().cpu()),
            "enzyme": float(s[self.ENZYME].detach().cpu()),
            "ph": float(s[self.PH].detach().cpu()),
            "t": int(self.step_count[env_id].detach().cpu()),
        }


def uniform(device: torch.device, n: int, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.rand(n, device=device)


def michaelis_menten(x: torch.Tensor, km: torch.Tensor) -> torch.Tensor:
    return x / (km + x + 1e-8)


def add_relative_noise(x: torch.Tensor, rel_std: float) -> torch.Tensor:
    noise_scale = rel_std * torch.clamp(torch.abs(x), min=0.05)
    return x + torch.randn_like(x) * noise_scale


def mask_stop_logits_from_pre_step(
    logits: torch.Tensor,
    pre_step_count: torch.Tensor,
    horizon: int,
    min_stop_step: int,
) -> torch.Tensor:
    del horizon  # kept for signature symmetry with mask_stop_logits_from_obs
    masked = logits.clone()
    can_select_stop = (pre_step_count + 1) >= min_stop_step
    masked[:, -1] = torch.where(
        can_select_stop,
        masked[:, -1],
        torch.full_like(masked[:, -1], -1.0e9),
    )
    return masked


def mask_stop_logits_from_obs(
    logits: torch.Tensor,
    obs: torch.Tensor,
    horizon: int,
    min_stop_step: int,
) -> torch.Tensor:
    # The final observation feature is normalized current step_count / horizon.
    pre_step_count = torch.round(obs[:, -1] * horizon).long()
    return mask_stop_logits_from_pre_step(
        logits=logits,
        pre_step_count=pre_step_count,
        horizon=horizon,
        min_stop_step=min_stop_step,
    )


PolicyFn = Callable[[IVTVectorEnv], torch.Tensor]


def fixed_batch_policy(stop_step: int) -> PolicyFn:
    def policy(env: IVTVectorEnv) -> torch.Tensor:
        stop = env.step_count >= stop_step
        continue_action = env.feed_action_index(0, 0)
        stop_action = env.stop_action

        actions = torch.full(
            (env.num_envs,),
            continue_action,
            device=env.device,
            dtype=torch.long,
        )
        actions = torch.where(stop, torch.full_like(actions, stop_action), actions)
        return actions

    return policy


def fixed_fed_batch_policy(
    ntp_level: int,
    mg_level: int,
    feed_interval: int,
    stop_step: int,
) -> PolicyFn:
    def policy(env: IVTVectorEnv) -> torch.Tensor:
        should_feed = (env.step_count % feed_interval == 0) & (env.step_count < stop_step)
        should_stop = env.step_count >= stop_step

        ntp = torch.where(
            should_feed,
            torch.full_like(env.step_count, ntp_level),
            torch.zeros_like(env.step_count),
        )
        mg = torch.where(
            should_feed,
            torch.full_like(env.step_count, mg_level),
            torch.zeros_like(env.step_count),
        )
        stop = should_stop.long()

        feed_actions = (ntp * 3 + mg).long()
        return torch.where(should_stop, torch.full_like(feed_actions, env.stop_action), feed_actions)

    return policy


def threshold_policy(
    ntp_threshold: float,
    mg_threshold: float,
    ntp_level: int,
    mg_level: int,
    max_step: int,
    ppi_stop: float = 3.50,
    enzyme_stop: float = 0.25,
    mgppi_stop: float = 2.50,
) -> PolicyFn:
    def policy(env: IVTVectorEnv) -> torch.Tensor:
        s = env.state

        ntp_low = s[:, env.NTP] < ntp_threshold
        mg_low = s[:, env.MG] < mg_threshold

        ntp = torch.where(
            ntp_low,
            torch.full_like(env.step_count, ntp_level),
            torch.zeros_like(env.step_count),
        )
        mg = torch.where(
            mg_low,
            torch.full_like(env.step_count, mg_level),
            torch.zeros_like(env.step_count),
        )

        should_stop = (
            (env.step_count >= max_step)
            | ((s[:, env.PPI] > ppi_stop) & (s[:, env.ENZYME] < enzyme_stop))
            | (s[:, env.MG_PPI] > mgppi_stop)
        )

        stop = should_stop.long()
        feed_actions = (ntp * 3 + mg).long()
        return torch.where(should_stop, torch.full_like(feed_actions, env.stop_action), feed_actions)

    return policy


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 64) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


@torch.no_grad()
def evaluate_policy(
    cfg: EnvConfig,
    policy_fn: PolicyFn,
    device: torch.device,
    num_episodes: int = 1000,
    num_envs: int = 256,
    seed: int = 123,
) -> Dict[str, float]:
    set_global_seeds(seed)

    env = IVTVectorEnv(num_envs=num_envs, cfg=cfg, device=device, seed=seed)
    env.reset()

    completed = 0

    records: Dict[str, List[float]] = {
        "qa_yield": [],
        "rna_full": [],
        "rna_trunc": [],
        "integrity": [],
        "ppi": [],
        "mg_ppi": [],
        "ph": [],
        "enzyme": [],
        "total_reward": [],
        "total_ntp_fed": [],
        "total_mg_fed": [],
        "stop_time": [],
    }

    while completed < num_episodes:
        actions = policy_fn(env)
        _, _, done, info = env.step(actions)

        if done.any():
            done_ids = torch.where(done)[0]
            remaining = num_episodes - completed
            record_ids = done_ids[:remaining]

            for key in records:
                values = info[key][record_ids].detach().cpu().numpy().tolist()
                records[key].extend(values)

            completed += len(record_ids)
            env.reset(done_ids)

    out: Dict[str, float] = {}
    for key, vals in records.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    return out


def ppo_policy_fn(model: ActorCritic, deterministic: bool = True) -> PolicyFn:
    @torch.no_grad()
    def policy(env: IVTVectorEnv) -> torch.Tensor:
        obs = env.get_obs()
        logits, _ = model(obs)
        logits = env.mask_action_logits(logits)

        if deterministic:
            return torch.argmax(logits, dim=-1)

        dist = Categorical(logits=logits)
        return dist.sample()

    return policy


def search_baselines(
    cfg: EnvConfig,
    device: torch.device,
    search_episodes: int = 256,
    eval_envs: int = 256,
    seed: int = 100,
) -> Dict[str, Tuple[str, PolicyFn, Dict[str, float]]]:
    candidates: Dict[str, List[Tuple[str, PolicyFn]]] = {
        "fixed_batch": [],
        "fixed_fed_batch": [],
        "threshold": [],
    }

    # Stop-time grid starts at min_stop_step because earlier stop requests are ignored.
    for stop_step in [cfg.min_stop_step, 52, 56, 60, 64]:
        name = f"fixed_batch_stop={stop_step}"
        candidates["fixed_batch"].append((name, fixed_batch_policy(stop_step)))

    # Fixed schedule grid includes Mg now that Mg limitation is more important.
    for ntp_level in [1, 2]:
        for mg_level in [0, 1, 2]:
            for interval in [4, 8, 12, 16]:
                for stop_step in [cfg.min_stop_step, 52, 56, 60, 64]:
                    name = (
                        f"fixed_feed_ntp={ntp_level}_mg={mg_level}"
                        f"_interval={interval}_stop={stop_step}"
                    )
                    candidates["fixed_fed_batch"].append(
                        (
                            name,
                            fixed_fed_batch_policy(
                                ntp_level=ntp_level,
                                mg_level=mg_level,
                                feed_interval=interval,
                                stop_step=stop_step,
                            ),
                        )
                    )

    for ntp_threshold in [0.20, 0.35, 0.50, 0.65, 0.80]:
        for mg_threshold in [0.15, 0.30, 0.45, 0.60]:
            for ntp_level in [1, 2]:
                for mg_level in [1, 2]:
                    for max_step in [cfg.min_stop_step, 52, 56, 60, 64]:
                        name = (
                            f"threshold_ntp={ntp_threshold:.2f}_mg={mg_threshold:.2f}"
                            f"_ntp_level={ntp_level}_mg_level={mg_level}_max={max_step}"
                        )
                        candidates["threshold"].append(
                            (
                                name,
                                threshold_policy(
                                    ntp_threshold=ntp_threshold,
                                    mg_threshold=mg_threshold,
                                    ntp_level=ntp_level,
                                    mg_level=mg_level,
                                    max_step=max_step,
                                ),
                            )
                        )

    best: Dict[str, Tuple[str, PolicyFn, Dict[str, float]]] = {}

    for group, group_candidates in candidates.items():
        print(f"\nSearching {group}: {len(group_candidates)} candidates")

        best_score = -float("inf")
        best_tuple: Optional[Tuple[str, PolicyFn, Dict[str, float]]] = None

        for i, (name, policy) in enumerate(group_candidates, start=1):
            metrics = evaluate_policy(
                cfg=cfg,
                policy_fn=policy,
                device=device,
                num_episodes=search_episodes,
                num_envs=eval_envs,
                seed=seed,
            )

            score = metrics["qa_yield_mean"]

            if score > best_score:
                best_score = score
                best_tuple = (name, policy, metrics)

            if i % 40 == 0 or i == len(group_candidates):
                print(
                    f"  {i:4d}/{len(group_candidates)} searched; "
                    f"best qa_yield={best_score:.4f}"
                )

        assert best_tuple is not None
        best[group] = best_tuple

        name, _, metrics = best_tuple
        print(f"Best {group}: {name}")
        print_metrics(metrics)

    return best


def train_ppo(
    cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    device: torch.device,
    seed: int = 0,
) -> ActorCritic:
    set_global_seeds(seed)

    env = IVTVectorEnv(num_envs=ppo_cfg.num_envs, cfg=cfg, device=device, seed=seed)
    obs = env.reset()

    model = ActorCritic(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        hidden_size=ppo_cfg.hidden_size,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5)

    batch_size = ppo_cfg.num_envs * ppo_cfg.rollout_steps
    minibatch_size = batch_size // ppo_cfg.minibatches
    num_updates = ppo_cfg.total_steps // batch_size

    if num_updates < 1:
        raise ValueError(
            "total_steps must be at least num_envs * rollout_steps. "
            f"Got total_steps={ppo_cfg.total_steps}, batch_size={batch_size}."
        )

    print("\nPPO training")
    print(f"  device: {device}")
    print(f"  num_envs: {ppo_cfg.num_envs}")
    print(f"  rollout_steps: {ppo_cfg.rollout_steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  minibatch_size: {minibatch_size}")
    print(f"  num_updates: {num_updates}")
    print(f"  total_steps_actual: {num_updates * batch_size}")

    for update in range(1, num_updates + 1):
        obs_buf = torch.zeros(
            (ppo_cfg.rollout_steps, ppo_cfg.num_envs, env.obs_dim),
            device=device,
        )
        actions_buf = torch.zeros(
            (ppo_cfg.rollout_steps, ppo_cfg.num_envs),
            dtype=torch.long,
            device=device,
        )
        logprobs_buf = torch.zeros(
            (ppo_cfg.rollout_steps, ppo_cfg.num_envs),
            device=device,
        )
        rewards_buf = torch.zeros(
            (ppo_cfg.rollout_steps, ppo_cfg.num_envs),
            device=device,
        )
        dones_buf = torch.zeros(
            (ppo_cfg.rollout_steps, ppo_cfg.num_envs),
            device=device,
        )
        values_buf = torch.zeros(
            (ppo_cfg.rollout_steps, ppo_cfg.num_envs),
            device=device,
        )

        rollout_rewards = []

        for step in range(ppo_cfg.rollout_steps):
            with torch.no_grad():
                logits, value = model(obs)
                logits = env.mask_action_logits(logits)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            next_obs, rewards, dones, _info = env.step(actions)

            obs_buf[step] = obs
            actions_buf[step] = actions
            logprobs_buf[step] = logprobs
            rewards_buf[step] = rewards
            dones_buf[step] = dones.float()
            values_buf[step] = value

            rollout_rewards.append(float(rewards.mean().detach().cpu()))

            if dones.any():
                done_ids = torch.where(dones)[0]
                env.reset(done_ids)
                next_obs = env.get_obs()

            obs = next_obs

        with torch.no_grad():
            _, next_value = model(obs)

        advantages = torch.zeros_like(rewards_buf, device=device)
        last_gae = torch.zeros(ppo_cfg.num_envs, device=device)

        for t in reversed(range(ppo_cfg.rollout_steps)):
            if t == ppo_cfg.rollout_steps - 1:
                next_nonterminal = 1.0 - dones_buf[t]
                next_values = next_value
            else:
                next_nonterminal = 1.0 - dones_buf[t]
                next_values = values_buf[t + 1]

            delta = (
                rewards_buf[t]
                + ppo_cfg.gamma * next_values * next_nonterminal
                - values_buf[t]
            )
            last_gae = (
                delta
                + ppo_cfg.gamma * ppo_cfg.gae_lambda * next_nonterminal * last_gae
            )
            advantages[t] = last_gae

        returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1, env.obs_dim))
        b_actions = actions_buf.reshape(-1)
        b_old_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_advantages = (
            b_advantages - b_advantages.mean()
        ) / (b_advantages.std() + 1e-8)

        indices = torch.arange(batch_size, device=device)

        approx_kl = torch.tensor(0.0, device=device)
        clip_fraction = torch.tensor(0.0, device=device)

        for _epoch in range(ppo_cfg.ppo_epochs):
            perm = indices[torch.randperm(batch_size, device=device)]

            for start in range(0, batch_size, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]

                logits, new_values = model(b_obs[mb_idx])
                logits = mask_stop_logits_from_obs(
                    logits=logits,
                    obs=b_obs[mb_idx],
                    horizon=cfg.horizon,
                    min_stop_step=cfg.min_stop_step,
                )
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(b_actions[mb_idx])
                entropy = dist.entropy().mean()

                log_ratio = new_logprobs - b_old_logprobs[mb_idx]
                ratio = log_ratio.exp()

                mb_adv = b_advantages[mb_idx]

                pg_loss_unclipped = -mb_adv * ratio
                pg_loss_clipped = -mb_adv * torch.clamp(
                    ratio,
                    1.0 - ppo_cfg.clip_eps,
                    1.0 + ppo_cfg.clip_eps,
                )
                policy_loss = torch.max(pg_loss_unclipped, pg_loss_clipped).mean()

                value_loss = 0.5 * (new_values - b_returns[mb_idx]).pow(2).mean()

                loss = (
                    policy_loss
                    + ppo_cfg.value_coef * value_loss
                    - ppo_cfg.entropy_coef * entropy
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        ((ratio - 1.0).abs() > ppo_cfg.clip_eps).float().mean()
                    )

        if update == 1 or update % max(1, num_updates // 10) == 0 or update == num_updates:
            print(
                f"update {update:4d}/{num_updates} | "
                f"mean_rollout_reward={np.mean(rollout_rewards): .4f} | "
                f"approx_kl={float(approx_kl.detach().cpu()):.4f} | "
                f"clip_frac={float(clip_fraction.detach().cpu()):.3f}"
            )

    return model


@torch.no_grad()
def collect_trajectory(
    cfg: EnvConfig,
    policy_fn: PolicyFn,
    device: torch.device,
    seed: int = 999,
) -> Tuple[List[Dict[str, float]], List[int]]:
    set_global_seeds(seed)

    env = IVTVectorEnv(num_envs=1, cfg=cfg, device=device, seed=seed)
    env.reset()

    traj = [env.raw_state_dict(0)]
    actions_taken: List[int] = []

    done = torch.tensor([False], device=device)
    while not bool(done[0].item()):
        action = policy_fn(env)
        actions_taken.append(int(action[0].detach().cpu()))
        _, _, done, _ = env.step(action)
        traj.append(env.raw_state_dict(0))

    return traj, actions_taken


def plot_trajectories(
    trajectories: Dict[str, List[Dict[str, float]]],
    out_path: str,
) -> None:
    metrics = [
        "rna_full",
        "rna_trunc",
        "ntp",
        "mg_free",
        "ppi",
        "mg_ppi",
        "enzyme",
        "ph",
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for name, traj in trajectories.items():
            x = [row["t"] for row in traj]
            y = [row[metric] for row in traj]
            ax.plot(x, y, label=name)

        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_action_trace(
    actions_by_policy: Dict[str, List[int]],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for name, actions in actions_by_policy.items():
        t = np.arange(len(actions))
        decoded = np.asarray([IVTVectorEnv.decode_action_index(a) for a in actions])
        ntp = decoded[:, 0]
        mg = decoded[:, 1]
        stop = decoded[:, 2]

        axes[0].step(t, ntp, where="post", label=name)
        axes[1].step(t, mg, where="post", label=name)
        axes[2].step(t, stop, where="post", label=name)

    axes[0].set_ylabel("NTP feed level")
    axes[1].set_ylabel("Mg feed level")
    axes[2].set_ylabel("Stop")
    axes[2].set_xlabel("step")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    axes[0].set_yticks([0, 1, 2])
    axes[1].set_yticks([0, 1, 2])
    axes[2].set_yticks([0, 1])

    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(name)


def print_metrics(metrics: Dict[str, float]) -> None:
    keys = [
        "qa_yield_mean",
        "rna_full_mean",
        "integrity_mean",
        "total_ntp_fed_mean",
        "total_mg_fed_mean",
        "ppi_mean",
        "mg_ppi_mean",
        "stop_time_mean",
        "total_reward_mean",
    ]

    for key in keys:
        if key in metrics:
            print(f"  {key:24s}: {metrics[key]:.4f}")


def save_results_csv(rows: List[Dict[str, float | str]], out_path: str) -> None:
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["smoke", "baselines", "train", "all"],
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="ivt_experiment2_outputs")

    parser.add_argument(
        "--obs-mode",
        type=str,
        default="oracle",
        choices=["oracle", "cheap_pat", "delayed_atline", "noisy_missing"],
    )
    parser.add_argument("--assay-interval", type=int, default=8)
    parser.add_argument("--assay-noise", type=float, default=0.03)
    parser.add_argument("--missing-prob", type=float, default=0.10)

    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--baseline-search-episodes", type=int, default=256)
    parser.add_argument("--eval-envs", type=int, default=256)

    # Sweep-ready PPO hyperparameters.
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.010)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--clip-eps", type=float, default=0.20)
    parser.add_argument("--gae-lambda", type=float, default=0.95)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device(args.device)
    set_global_seeds(args.seed)

    env_cfg = EnvConfig(
        obs_mode=args.obs_mode,
        assay_interval=args.assay_interval,
        assay_noise=args.assay_noise,
        missing_prob=args.missing_prob,
    )
    ppo_cfg = PPOConfig(
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        hidden_size=args.hidden_size,
        ppo_epochs=args.ppo_epochs,
        minibatches=args.minibatches,
        clip_eps=args.clip_eps,
        gae_lambda=args.gae_lambda,
    )

    print("Environment config:")
    for key, value in asdict(env_cfg).items():
        print(f"  {key}: {value}")

    print("\nRun config:")
    print(f"  mode: {args.mode}")
    print(f"  device: {device}")
    print(f"  seed: {args.seed}")
    print(f"  out_dir: {args.out_dir}")

    print("\nPPO config:")
    for key, value in asdict(ppo_cfg).items():
        print(f"  {key}: {value}")

    if args.mode == "smoke":
        print("\nRunning smoke test.")

        env = IVTVectorEnv(num_envs=4, cfg=env_cfg, device=device, seed=args.seed)
        obs = env.reset()

        print(f"obs shape: {tuple(obs.shape)}")

        for step in range(5):
            actions = torch.randint(0, env.n_actions, (4,), device=device)
            obs, reward, done, info = env.step(actions)

            print(
                f"step={step} "
                f"reward={reward.detach().cpu().numpy()} "
                f"done={done.detach().cpu().numpy()} "
                f"qa={info['qa_yield'].detach().cpu().numpy()}"
            )

        print("Smoke test complete.")
        return

    results: List[Dict[str, float | str]] = []
    best_baselines: Dict[str, Tuple[str, PolicyFn, Dict[str, float]]] = {}

    if args.mode in ["baselines", "all"]:
        best_baselines = search_baselines(
            cfg=env_cfg,
            device=device,
            search_episodes=args.baseline_search_episodes,
            eval_envs=args.eval_envs,
            seed=100,
        )

        print("\nFinal baseline evaluation")

        for group, (name, policy, _search_metrics) in best_baselines.items():
            metrics = evaluate_policy(
                cfg=env_cfg,
                policy_fn=policy,
                device=device,
                num_episodes=args.eval_episodes,
                num_envs=args.eval_envs,
                seed=999,
            )

            print(f"\n{group}: {name}")
            print_metrics(metrics)

            row: Dict[str, float | str] = {
                "method": group,
                "policy": name,
                "obs_mode": env_cfg.obs_mode,
            }
            row.update(metrics)
            results.append(row)

    model: Optional[ActorCritic] = None

    if args.mode in ["train", "all"]:
        model = train_ppo(
            cfg=env_cfg,
            ppo_cfg=ppo_cfg,
            device=device,
            seed=args.seed,
        )

        model_path = os.path.join(args.out_dir, "ppo_ivt_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"\nSaved PPO model to: {model_path}")

        ppo_eval = evaluate_policy(
            cfg=env_cfg,
            policy_fn=ppo_policy_fn(model, deterministic=True),
            device=device,
            num_episodes=args.eval_episodes,
            num_envs=args.eval_envs,
            seed=999,
        )

        print("\nPPO final evaluation")
        print_metrics(ppo_eval)

        row = {
            "method": "ppo",
            "policy": "discrete_mlp_ppo",
            "obs_mode": env_cfg.obs_mode,
        }
        row.update(ppo_eval)
        results.append(row)

    if results:
        csv_path = os.path.join(args.out_dir, "experiment1_results.csv")
        save_results_csv(results, csv_path)
        print(f"\nSaved results to: {csv_path}")

    trajectories: Dict[str, List[Dict[str, float]]] = {}
    actions_by_policy: Dict[str, List[int]] = {}

    if best_baselines:
        for group, (_name, policy, _metrics) in best_baselines.items():
            traj, actions = collect_trajectory(
                cfg=env_cfg,
                policy_fn=policy,
                device=device,
                seed=2024,
            )
            trajectories[group] = traj
            actions_by_policy[group] = actions

    if model is not None:
        traj, actions = collect_trajectory(
            cfg=env_cfg,
            policy_fn=ppo_policy_fn(model, deterministic=True),
            device=device,
            seed=2024,
        )
        trajectories["ppo"] = traj
        actions_by_policy["ppo"] = actions

    if trajectories:
        plot_path = os.path.join(args.out_dir, "representative_trajectories.png")
        plot_trajectories(trajectories, plot_path)
        print(f"Saved representative trajectory plot to: {plot_path}")

    if actions_by_policy:
        action_plot_path = os.path.join(args.out_dir, "representative_action_trace.png")
        plot_action_trace(actions_by_policy, action_plot_path)
        print(f"Saved representative action trace to: {action_plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
