from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

from ac_pbgrl.config import Config
from ac_pbgrl.models.policy import ACPolicyNetwork, PrivilegedQNetwork
from ac_pbgrl.state import PotentialSupervisionBatch, TransitionBatch

from .distributed import BatchSchedule, DistributedContext, maybe_no_sync, unwrap, wrap_ddp
from .losses import (
    auxiliary_weight,
    heteroscedastic_gaussian_nll,
    masked_regression_loss,
    ranknet_loss,
)


@dataclass
class SACLearnerState:
    update_step: int = 0
    target_counter: int = 0
    environment_steps: int = 0
    episodes: int = 0
    update_credit: float = 0.0


class DiscreteSACLearner:
    def __init__(self, config: Config, context: DistributedContext) -> None:
        self.config = config
        self.context = context
        method = config.method
        model = config.model
        environment = config.environment
        self.actor_raw = ACPolicyNetwork(
            node_feature_dim=int(environment.node_feature_dim),
            edge_feature_dim=int(environment.edge_feature_dim),
            embedding_dim=int(model.embedding_dim),
            heads=int(model.heads),
            layers=int(model.encoder_layers),
            dropout=float(model.dropout),
            use_potential=bool(method.potential),
            use_diffusion=bool(method.graph_diffusion),
            fuse_uncertainty=bool(method.fuse_uncertainty),
            logvar_min=float(model.logvar_min),
            logvar_max=float(model.logvar_max),
        ).to(context.device)
        critic_dim = int(environment.critic_feature_dim if method.privileged_critic else environment.node_feature_dim)
        self.q1_raw = PrivilegedQNetwork(
            critic_dim,
            int(environment.edge_feature_dim),
            int(model.embedding_dim),
            int(model.heads),
            int(model.encoder_layers),
            float(model.dropout),
            use_diffusion=False,
        ).to(context.device)
        self.q2_raw = copy.deepcopy(self.q1_raw).to(context.device)
        self.target_q1 = copy.deepcopy(self.q1_raw).to(context.device).eval()
        self.target_q2 = copy.deepcopy(self.q2_raw).to(context.device).eval()
        self.actor = wrap_ddp(self.actor_raw, context)
        self.q1 = wrap_ddp(self.q1_raw, context)
        self.q2 = wrap_ddp(self.q2_raw, context)
        learning_rate = float(config.train.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor_raw.parameters(), lr=learning_rate)
        self.q1_optimizer = torch.optim.Adam(self.q1_raw.parameters(), lr=learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q2_raw.parameters(), lr=learning_rate)
        self.log_alpha = nn.Parameter(torch.zeros((), device=context.device))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(config.train.alpha_learning_rate))
        candidate_count = int(environment.candidate_padding)
        self.target_entropy = 0.05 * (-np.log(1.0 / candidate_count))
        self.state = SACLearnerState()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _critic_graph(self, batch: TransitionBatch, next_state: bool = False):
        if self.config.method.privileged_critic:
            graph = batch.critic_next_state if next_state else batch.critic_state
            if graph is None:
                raise ValueError("privileged critic is enabled but critic graph is missing")
            return graph
        return batch.next_state if next_state else batch.state

    def _actor_loss(self, batch: TransitionBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.actor(batch.state)
        critic_state = self._critic_graph(batch)
        with torch.no_grad():
            q_values = torch.minimum(self.q1(critic_state), self.q2(critic_state))
        objective = output.probabilities * (self.alpha.detach() * output.log_probs - q_values.detach())
        policy_loss = objective.sum(dim=-1).mean()
        metrics: dict[str, torch.Tensor] = {"loss/policy_sac": policy_loss.detach()}

        potential_weight = auxiliary_weight(
            self.state.update_step,
            float(self.config.loss.potential_weight),
            int(self.config.loss.warmup_steps),
            int(self.config.loss.ramp_steps),
        )
        if self.config.method.potential and batch.future_gain is not None and batch.future_gain_mask is not None:
            label_mask = batch.future_gain_mask & batch.state.candidate_mask & torch.isfinite(batch.future_gain)
            if self.config.method.potential_loss == "nll":
                potential_loss = heteroscedastic_gaussian_nll(
                    output.action_mean,
                    output.action_log_variance,
                    batch.future_gain,
                    label_mask,
                    logvar_min=float(self.config.model.logvar_min),
                    logvar_max=float(self.config.model.logvar_max),
                )
                region_loss = heteroscedastic_gaussian_nll(
                    output.region_mean,
                    output.region_log_variance,
                    batch.future_gain,
                    label_mask,
                    logvar_min=float(self.config.model.logvar_min),
                    logvar_max=float(self.config.model.logvar_max),
                )
            else:
                potential_loss = masked_regression_loss(
                    output.action_mean, batch.future_gain, label_mask, kind=str(self.config.method.potential_loss)
                )
                region_loss = masked_regression_loss(
                    output.region_mean,
                    batch.future_gain,
                    label_mask,
                    kind=str(self.config.method.potential_loss),
                )
            region_weight = float(self.config.loss.get("region_potential_weight", 0.5))
            policy_loss = policy_loss + potential_weight * (potential_loss + region_weight * region_loss)
            metrics["loss/potential"] = potential_loss.detach()
            metrics["loss/region_potential"] = region_loss.detach()
            if self.config.method.ranknet:
                valid_targets = batch.future_gain[label_mask]
                target_std = valid_targets.std(unbiased=False) if valid_targets.numel() else output.logits.new_zeros(())
                tie_delta = float(self.config.loss.rank_tie_delta_std) * float(target_std.detach())
                ranking = ranknet_loss(
                    output.action_mean,
                    batch.future_gain,
                    label_mask,
                    tie_delta=tie_delta,
                    max_pairs_per_state=int(self.config.loss.rank_pairs_per_state),
                )
                rank_weight = auxiliary_weight(
                    self.state.update_step,
                    float(self.config.loss.rank_weight),
                    int(self.config.loss.warmup_steps),
                    int(self.config.loss.ramp_steps),
                )
                policy_loss = policy_loss + rank_weight * ranking
                metrics["loss/ranknet"] = ranking.detach()
        if self.config.method.q_distillation and batch.teacher_q is not None:
            mask = batch.state.candidate_mask & torch.isfinite(batch.teacher_q)
            centered_teacher = batch.teacher_q - batch.teacher_q.masked_fill(~mask, 0).sum(1, keepdim=True) / mask.sum(
                1, keepdim=True
            ).clamp_min(1)
            centered_logits = output.base_logits - output.base_logits.masked_fill(~mask, 0).sum(
                1, keepdim=True
            ) / mask.sum(1, keepdim=True).clamp_min(1)
            distillation = masked_regression_loss(centered_logits, centered_teacher, mask, kind="mse")
            policy_loss = policy_loss + float(self.config.loss.q_distill_weight) * distillation
            metrics["loss/q_distillation"] = distillation.detach()
        entropy = -(output.probabilities * output.log_probs).sum(dim=-1).mean()
        metrics["policy/entropy"] = entropy.detach()
        metrics["policy/auxiliary_weight"] = output.logits.new_tensor(potential_weight)
        return policy_loss, metrics

    def _critic_losses(self, batch: TransitionBatch) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        critic_state = self._critic_graph(batch)
        critic_next = self._critic_graph(batch, next_state=True)
        with torch.no_grad():
            next_output = self.actor(batch.next_state)
            next_q = torch.minimum(self.target_q1(critic_next), self.target_q2(critic_next))
            next_value = (
                next_output.probabilities
                * (next_q - self.alpha.detach() * next_output.log_probs)
            ).sum(dim=-1)
            target = batch.reward.reshape(-1) + float(self.config.train.gamma) * (
                1.0 - batch.done.reshape(-1)
            ) * next_value
        action = batch.action.reshape(-1, 1).long()
        q1 = torch.gather(self.q1(critic_state), 1, action).squeeze(1)
        q2 = torch.gather(self.q2(critic_state), 1, action).squeeze(1)
        q1_loss = torch.nn.functional.mse_loss(q1.float(), target.float())
        q2_loss = torch.nn.functional.mse_loss(q2.float(), target.float())
        return q1_loss, q2_loss, {"loss/q1": q1_loss.detach(), "loss/q2": q2_loss.detach(), "value/target": target.mean()}

    def _potential_loss(
        self, batch: PotentialSupervisionBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.actor(batch.state)
        mask = batch.future_gain_mask & batch.state.candidate_mask & torch.isfinite(batch.future_gain)
        if str(self.config.method.potential_loss) == "nll":
            regression = heteroscedastic_gaussian_nll(
                output.action_mean,
                output.action_log_variance,
                batch.future_gain,
                mask,
                logvar_min=float(self.config.model.logvar_min),
                logvar_max=float(self.config.model.logvar_max),
            )
            region_regression = heteroscedastic_gaussian_nll(
                output.region_mean,
                output.region_log_variance,
                batch.future_gain,
                mask,
                logvar_min=float(self.config.model.logvar_min),
                logvar_max=float(self.config.model.logvar_max),
            )
        else:
            regression = masked_regression_loss(
                output.action_mean,
                batch.future_gain,
                mask,
                kind=str(self.config.method.potential_loss),
            )
            region_regression = masked_regression_loss(
                output.region_mean,
                batch.future_gain,
                mask,
                kind=str(self.config.method.potential_loss),
            )
        potential_weight = auxiliary_weight(
            self.state.update_step,
            float(self.config.loss.potential_weight),
            int(self.config.loss.warmup_steps),
            int(self.config.loss.ramp_steps),
        )
        region_weight = float(self.config.loss.get("region_potential_weight", 0.5))
        total = potential_weight * (regression + region_weight * region_regression)
        metrics = {
            "loss/potential_offline": regression.detach(),
            "loss/region_potential_offline": region_regression.detach(),
            "policy/auxiliary_weight": output.logits.new_tensor(potential_weight),
        }
        if bool(self.config.method.ranknet):
            valid_targets = batch.future_gain[mask]
            target_std = valid_targets.std(unbiased=False) if valid_targets.numel() else output.logits.new_zeros(())
            ranking = ranknet_loss(
                output.action_mean,
                batch.future_gain,
                mask,
                tie_delta=float(self.config.loss.rank_tie_delta_std) * float(target_std.detach()),
                max_pairs_per_state=int(self.config.loss.rank_pairs_per_state),
            )
            rank_weight = auxiliary_weight(
                self.state.update_step,
                float(self.config.loss.rank_weight),
                int(self.config.loss.warmup_steps),
                int(self.config.loss.ramp_steps),
            )
            total = total + rank_weight * ranking
            metrics["loss/ranknet_offline"] = ranking.detach()
        return total, metrics

    def update_chunks(
        self,
        chunks: list[TransitionBatch],
        schedule: BatchSchedule,
        potential_chunks: list[PotentialSupervisionBatch] | None = None,
        potential_schedule: BatchSchedule | None = None,
    ) -> dict[str, float]:
        if len(chunks) != len(schedule.chunk_sizes):
            raise ValueError("chunk list does not match batch schedule")
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.q1_optimizer.zero_grad(set_to_none=True)
        self.q2_optimizer.zero_grad(set_to_none=True)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        valid_chunks = [(batch, size) for batch, size in zip(chunks, schedule.chunk_sizes) if size > 0]
        for batch, chunk_size in valid_chunks:
            batch = batch.to(self.context.device)
            scale = schedule.gradient_scale(chunk_size)
            use_bf16 = bool(self.config.train.bf16) and self.context.device.type == "cuda"
            # A single explicit reduction after both RL and privileged-label
            # batches keeps the update invariant to DDP world size.
            with maybe_no_sync((self.actor, self.q1, self.q2), enabled=self.context.initialized):
                with torch.autocast(device_type=self.context.device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    actor_loss, actor_metrics = self._actor_loss(batch)
                (actor_loss * scale).backward()
                with torch.autocast(device_type=self.context.device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    q1_loss, q2_loss, critic_metrics = self._critic_losses(batch)
                (q1_loss * scale).backward()
                (q2_loss * scale).backward()
                output = self.actor(batch.state)
                entropy_term = (output.probabilities * output.log_probs).sum(dim=-1).detach()
                alpha_loss = -(self.log_alpha * (entropy_term + self.target_entropy)).mean()
                (alpha_loss * scale).backward()
            for name, value in {**actor_metrics, **critic_metrics, "loss/alpha": alpha_loss.detach()}.items():
                totals[name] = totals.get(name, 0.0) + float(value) * (chunk_size / schedule.local_samples)

        if potential_chunks is not None:
            if potential_schedule is None or len(potential_chunks) != len(potential_schedule.chunk_sizes):
                raise ValueError("potential chunks require a matching batch schedule")
            for batch, chunk_size in zip(potential_chunks, potential_schedule.chunk_sizes):
                if chunk_size <= 0:
                    continue
                batch = batch.to(self.context.device)
                scale = potential_schedule.gradient_scale(chunk_size)
                use_bf16 = bool(self.config.train.bf16) and self.context.device.type == "cuda"
                with maybe_no_sync((self.actor,), enabled=self.context.initialized):
                    with torch.autocast(
                        device_type=self.context.device.type,
                        dtype=torch.bfloat16,
                        enabled=use_bf16,
                    ):
                        auxiliary_loss, auxiliary_metrics = self._potential_loss(batch)
                    (auxiliary_loss * scale).backward()
                for name, value in auxiliary_metrics.items():
                    totals[name] = totals.get(name, 0.0) + float(value) * (
                        chunk_size / potential_schedule.local_samples
                    )

        self.context.all_reduce_gradients(
            (self.actor_raw, self.q1_raw, self.q2_raw),
            extra_parameters=(self.log_alpha,),
        )

        actor_norm = torch.nn.utils.clip_grad_norm_(
            self.actor_raw.parameters(), float(self.config.train.max_grad_norm_actor)
        )
        q1_norm = torch.nn.utils.clip_grad_norm_(
            self.q1_raw.parameters(), float(self.config.train.max_grad_norm_critic)
        )
        q2_norm = torch.nn.utils.clip_grad_norm_(
            self.q2_raw.parameters(), float(self.config.train.max_grad_norm_critic)
        )
        if not all(torch.isfinite(value) for value in (actor_norm, q1_norm, q2_norm)):
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.q1_optimizer.zero_grad(set_to_none=True)
            self.q2_optimizer.zero_grad(set_to_none=True)
            self.alpha_optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite gradient detected; optimizer step was skipped")
        self.actor_optimizer.step()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        self.alpha_optimizer.step()
        self.state.update_step += 1
        self.state.target_counter += 1
        if self.state.target_counter >= int(self.config.train.target_update_interval):
            self.target_q1.load_state_dict(self.q1_raw.state_dict())
            self.target_q2.load_state_dict(self.q2_raw.state_dict())
            self.state.target_counter = 0
        sync_interval = int(self.config.train.get("ddp_sync_check_interval", 100))
        if sync_interval > 0 and self.state.update_step % sync_interval == 0:
            self.context.assert_parameters_synchronized(
                (self.actor_raw, self.q1_raw, self.q2_raw),
                extra_parameters=(self.log_alpha,),
            )
        totals = {
            name: self.context.weighted_metric_mean(value, schedule.local_samples)
            for name, value in totals.items()
        }
        totals["grad/actor"] = float(actor_norm)
        totals["grad/q"] = float(max(q1_norm, q2_norm))
        totals["policy/alpha"] = float(self.alpha.detach())
        totals["train/update_step"] = float(self.state.update_step)
        return totals

    def state_dict(self) -> dict:
        return {
            "actor": self.actor_raw.state_dict(),
            "q1": self.q1_raw.state_dict(),
            "q2": self.q2_raw.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q1_optimizer": self.q1_optimizer.state_dict(),
            "q2_optimizer": self.q2_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "learner_state": vars(self.state),
        }

    def load_state_dict(self, payload: dict) -> None:
        self.actor_raw.load_state_dict(payload["actor"])
        self.q1_raw.load_state_dict(payload["q1"])
        self.q2_raw.load_state_dict(payload["q2"])
        self.target_q1.load_state_dict(payload.get("target_q1", payload["q1"]))
        self.target_q2.load_state_dict(payload.get("target_q2", payload["q2"]))
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.q1_optimizer.load_state_dict(payload["q1_optimizer"])
        self.q2_optimizer.load_state_dict(payload["q2_optimizer"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.log_alpha.device))
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        self.state = SACLearnerState(**payload.get("learner_state", {}))
