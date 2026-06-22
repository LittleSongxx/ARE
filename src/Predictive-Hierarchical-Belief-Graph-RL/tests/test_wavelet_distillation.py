from __future__ import annotations

import math
import unittest
from unittest import mock

import torch
import torch.nn as nn

from hpbg_rl import driver as driver_module
from hpbg_rl.driver import create_learner_state, train_step
from hpbg_rl.parameter import CRITIC_NODE_INPUT_DIM, K_SIZE, NODE_INPUT_DIM, RuntimeConfig
from hpbg_rl.wavelet_graph import decompose_graph_hidden, gather_by_index
from hpbg_rl.wavelet_losses import WaveletDistillationLoss, masked_l1_mean


def build_training_batch(batch_size=2, actor_nodes=6, critic_nodes=8):
    node_inputs = torch.randn(batch_size, actor_nodes, NODE_INPUT_DIM)
    critic_node_inputs = torch.randn(batch_size, critic_nodes, CRITIC_NODE_INPUT_DIM)

    node_padding_mask = torch.zeros(batch_size, 1, actor_nodes, dtype=torch.bool)
    critic_node_padding_mask = torch.zeros(batch_size, 1, critic_nodes, dtype=torch.bool)

    edge_mask = torch.zeros(batch_size, actor_nodes, actor_nodes, dtype=torch.bool)
    critic_edge_mask = torch.zeros(batch_size, critic_nodes, critic_nodes, dtype=torch.bool)

    current_index = torch.zeros(batch_size, 1, 1, dtype=torch.long)
    critic_current_index = torch.full((batch_size, 1, 1), 3, dtype=torch.long)

    actor_to_critic_index = torch.tensor(
        [
            [3, 0, 5, 2, 1, 4],
            [3, 0, 5, 2, 1, 4],
        ],
        dtype=torch.long,
    )[:batch_size]

    current_edge = torch.zeros(batch_size, K_SIZE, 1, dtype=torch.long)
    current_edge[:, :4, 0] = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    critic_current_edge = torch.gather(actor_to_critic_index, 1, current_edge.squeeze(-1))
    critic_current_edge[:, 4:] = 0
    critic_current_edge = critic_current_edge.unsqueeze(-1)

    edge_padding_mask = torch.ones(batch_size, 1, K_SIZE, dtype=torch.bool)
    edge_padding_mask[:, :, :4] = False
    critic_edge_padding_mask = edge_padding_mask.clone()

    next_node_inputs = node_inputs.clone()
    next_node_padding_mask = node_padding_mask.clone()
    next_edge_mask = edge_mask.clone()
    next_current_index = current_index.clone()
    next_current_edge = current_edge.clone()
    next_edge_padding_mask = edge_padding_mask.clone()

    critic_next_node_inputs = critic_node_inputs.clone()
    critic_next_node_padding_mask = critic_node_padding_mask.clone()
    critic_next_edge_mask = critic_edge_mask.clone()
    critic_next_current_index = critic_current_index.clone()
    critic_next_current_edge = critic_current_edge.clone()
    critic_next_edge_padding_mask = critic_edge_padding_mask.clone()

    action = torch.zeros(batch_size, 1, 1, dtype=torch.long)
    reward = torch.randn(batch_size, 1, 1)
    done = torch.zeros(batch_size, 1, 1)

    return {
        "node_inputs": node_inputs,
        "node_padding_mask": node_padding_mask,
        "edge_mask": edge_mask,
        "current_index": current_index,
        "current_edge": current_edge,
        "edge_padding_mask": edge_padding_mask,
        "action": action,
        "reward": reward,
        "done": done,
        "next_node_inputs": next_node_inputs,
        "next_node_padding_mask": next_node_padding_mask,
        "next_edge_mask": next_edge_mask,
        "next_current_index": next_current_index,
        "next_current_edge": next_current_edge,
        "next_edge_padding_mask": next_edge_padding_mask,
        "critic_node_inputs": critic_node_inputs,
        "critic_node_padding_mask": critic_node_padding_mask,
        "critic_edge_mask": critic_edge_mask,
        "critic_current_index": critic_current_index,
        "critic_current_edge": critic_current_edge,
        "critic_edge_padding_mask": critic_edge_padding_mask,
        "critic_next_node_inputs": critic_next_node_inputs,
        "critic_next_node_padding_mask": critic_next_node_padding_mask,
        "critic_next_edge_mask": critic_next_edge_mask,
        "critic_next_current_index": critic_next_current_index,
        "critic_next_current_edge": critic_next_current_edge,
        "critic_next_edge_padding_mask": critic_next_edge_padding_mask,
        "actor_to_critic_index": actor_to_critic_index,
    }


class CallRecorder(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.calls = []

    def forward(self, *args, **kwargs):
        self.calls.append(dict(kwargs))
        return self.module(*args, **kwargs)


class WaveletDistillationTests(unittest.TestCase):
    def test_target_critic_teacher_is_used_and_receives_no_gradient(self):
        runtime_config = RuntimeConfig(
            use_privileged_wavelet_distillation=True,
            wavelet_distill_weight=0.1,
            wavelet_distill_warmup_updates=0,
            wavelet_distill_ramp_updates=0,
        )
        learner_state = create_learner_state(runtime_config, torch.device("cpu"))
        online_q1_recorder = CallRecorder(learner_state.q_net1)
        target_q1_recorder = CallRecorder(learner_state.target_q_net1)
        learner_state.q_net1_wrapper = online_q1_recorder
        learner_state.target_q_net1_wrapper = target_q1_recorder

        batch = build_training_batch()
        train_step(batch, learner_state, runtime_config)

        self.assertTrue(any(call.get("return_hidden") for call in target_q1_recorder.calls))
        self.assertFalse(any(call.get("return_hidden") for call in online_q1_recorder.calls))
        self.assertTrue(any(param.grad is not None for param in learner_state.policy_net.parameters()))
        self.assertTrue(all(param.grad is None for param in learner_state.target_q_net1.parameters()))

    def test_teacher_decomposition_uses_full_privileged_graph_before_overlap_gather(self):
        runtime_config = RuntimeConfig(
            use_privileged_wavelet_distillation=True,
            wavelet_distill_weight=0.1,
            wavelet_distill_warmup_updates=0,
            wavelet_distill_ramp_updates=0,
        )
        learner_state = create_learner_state(runtime_config, torch.device("cpu"))
        batch = build_training_batch(batch_size=1)

        recorded_calls = []
        original_decompose = driver_module.decompose_graph_hidden

        def record_decompose(hidden, edge_mask, node_padding_mask, *args, **kwargs):
            recorded_calls.append((edge_mask.detach().clone(), node_padding_mask.detach().clone()))
            return original_decompose(hidden, edge_mask, node_padding_mask, *args, **kwargs)

        with mock.patch.object(driver_module, "decompose_graph_hidden", side_effect=record_decompose):
            train_step(batch, learner_state, runtime_config)

        self.assertGreaterEqual(len(recorded_calls), 2)
        self.assertTrue(torch.equal(recorded_calls[0][0], batch["edge_mask"]))
        self.assertTrue(torch.equal(recorded_calls[1][0], batch["critic_edge_mask"]))
        self.assertTrue(torch.equal(recorded_calls[1][1], batch["critic_node_padding_mask"]))

        full_hidden = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [10.0, 10.0],
                ]
            ]
        )
        full_edge_mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        actor_edge_mask = torch.zeros(1, 3, 3, dtype=torch.bool)
        full_node_padding_mask = torch.zeros(1, 1, 4, dtype=torch.bool)
        actor_node_padding_mask = torch.zeros(1, 1, 3, dtype=torch.bool)
        actor_to_critic_index = torch.tensor([[0, 1, 2]], dtype=torch.long)

        full_lf, _, *_ = decompose_graph_hidden(full_hidden, full_edge_mask, full_node_padding_mask, scales=(1,))
        full_overlap = gather_by_index(full_lf, actor_to_critic_index)

        actor_first_hidden = gather_by_index(full_hidden, actor_to_critic_index)
        actor_first_lf, _, *_ = decompose_graph_hidden(
            actor_first_hidden,
            actor_edge_mask,
            actor_node_padding_mask,
            scales=(1,),
        )

        self.assertFalse(torch.allclose(full_overlap, actor_first_lf))

    def test_overlap_valid_mask_controls_loss_and_warmup(self):
        distill = WaveletDistillationLoss(1.0, 1.0, 0.5, 10, 5)
        actor = torch.zeros(1, 4, 3)
        critic = actor.clone()
        actor[:, 2:] = 10.0
        critic[:, 2:] = -10.0
        overlap_valid_mask = torch.tensor([[True, True, False, False]])

        loss_dict = distill(actor, actor, critic, critic, overlap_valid_mask, update_step=0)
        self.assertEqual(loss_dict["lambda_eff"], 0.0)
        self.assertEqual(float(loss_dict["loss_lf"]), 0.0)
        self.assertEqual(float(loss_dict["loss_hf"]), 0.0)

    def test_masked_l1_mean_averages_over_nodes_and_features(self):
        mask = torch.tensor([[True, False]])
        small_prediction = torch.ones(1, 2, 4)
        large_prediction = torch.ones(1, 2, 64)
        small_target = torch.zeros_like(small_prediction)
        large_target = torch.zeros_like(large_prediction)

        small_loss = masked_l1_mean(small_prediction, small_target, valid_mask=mask)
        large_loss = masked_l1_mean(large_prediction, large_target, valid_mask=mask)

        self.assertAlmostEqual(float(small_loss), 1.0, places=6)
        self.assertAlmostEqual(float(large_loss), 1.0, places=6)

    def test_distillation_pipeline_remains_finite(self):
        runtime_config = RuntimeConfig(
            use_privileged_wavelet_distillation=True,
            wavelet_distill_weight=0.1,
            wavelet_distill_warmup_updates=0,
            wavelet_distill_ramp_updates=0,
        )
        learner_state = create_learner_state(runtime_config, torch.device("cpu"))
        batch = build_training_batch()

        metrics = train_step(batch, learner_state, runtime_config)

        for key in ("policy_loss", "q_value_loss", "alpha_loss", "wavelet_loss"):
            self.assertTrue(math.isfinite(float(metrics[key])), key)


if __name__ == "__main__":
    unittest.main()
