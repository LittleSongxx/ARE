import copy
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from ac_pbgrl.config import load_config
from ac_pbgrl.data.labels import LabelDataset, LabelShardWriter
from ac_pbgrl.envs.synthetic import SyntheticGraphExplorationEnv
from ac_pbgrl.learning.distributed import BatchSchedule, DistributedContext
from ac_pbgrl.learning.future_gain import FutureGainLabeler
from ac_pbgrl.learning.losses import heteroscedastic_gaussian_nll, ranknet_loss
from ac_pbgrl.learning.replay import PersistentReplayBuffer
from ac_pbgrl.learning.sac import DiscreteSACLearner
from ac_pbgrl.learning.teacher import HeuristicTeacher
from ac_pbgrl.models.context import action_preserving_context
from ac_pbgrl.models.policy import ACPolicyNetwork
from ac_pbgrl.models.temporal import GRUPotentialMemory
from ac_pbgrl.state import TransitionBatch


def smoke_config():
    from ac_pbgrl.learning.train import apply_smoke_overrides

    return apply_smoke_overrides(load_config("full", overrides=["project.data_root=.runtime-test"]))


def test_policy_shapes_masks_and_potential_outputs():
    config = smoke_config()
    environment = SyntheticGraphExplorationEnv(32, 8, seed=1)
    state, _ = environment.reset()
    policy = ACPolicyNetwork(4, 4, 32, 4, 2, use_potential=True, use_diffusion=True)
    output = policy(state)
    assert output.logits.shape == (1, 8)
    assert output.action_mean.shape == (1, 8)
    assert output.region_mean.shape == (1, 8)
    assert torch.all(output.action_log_variance >= output.region_log_variance)
    assert torch.isfinite(output.log_probs[state.candidate_mask]).all()
    assert torch.all(output.logits[~state.candidate_mask] < -1000)


def test_action_preserving_compaction_keeps_slot_order():
    environment = SyntheticGraphExplorationEnv(32, 8, seed=2)
    state, _ = environment.reset()
    original_ids = torch.gather(state.stable_ids, 1, state.candidate_indices)
    compact = action_preserving_context(state, local_budget=8, region_budget=4, region_size_m=8)
    compact_ids = torch.gather(compact.stable_ids, 1, compact.candidate_indices)
    assert torch.equal(original_ids[state.candidate_mask], compact_ids[compact.candidate_mask])
    assert compact.node_features.shape[1] == 12
    assert "a_star_skeletons" in compact.metadata["compaction"][0]


def test_future_gain_labels_all_valid_candidates():
    environment = SyntheticGraphExplorationEnv(32, 8, seed=3)
    state, _ = environment.reset()
    labels = FutureGainLabeler(HeuristicTeacher(), horizon=3, gamma=0.9).label(environment, state)
    assert np.array_equal(labels.mask, state.candidate_mask[0].numpy())
    assert np.isfinite(labels.values[labels.mask]).all()
    assert (labels.rollout_lengths[labels.mask] >= 1).all()


def test_losses_are_finite_and_rank_ties_are_ignored():
    mean = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    logvar = torch.tensor([[-100.0, 0.0, 100.0]], requires_grad=True)
    target = torch.tensor([[0.5, 1.0, 3.0]])
    mask = torch.tensor([[True, True, True]])
    nll = heteroscedastic_gaussian_nll(mean, logvar, target, mask)
    rank = ranknet_loss(mean, target, mask, tie_delta=0.1)
    (nll + rank).backward()
    assert torch.isfinite(nll)
    assert torch.isfinite(mean.grad).all()


def test_padded_nan_targets_do_not_poison_auxiliary_losses():
    mean = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    logvar = torch.zeros_like(mean, requires_grad=True)
    target = torch.tensor([[1.5, float("nan"), float("nan")]])
    mask = torch.tensor([[True, False, False]])

    nll = heteroscedastic_gaussian_nll(mean, logvar, target, mask)
    rank = ranknet_loss(mean, target, mask)
    (nll + rank).backward()

    assert torch.isfinite(nll)
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(logvar.grad).all()


def test_label_shards_feed_offline_potential_batch(tmp_path: Path):
    environment = SyntheticGraphExplorationEnv(32, 8, seed=31)
    state, _ = environment.reset()
    labels = FutureGainLabeler(HeuristicTeacher(), horizon=2, gamma=0.95).label(environment, state)
    with LabelShardWriter(tmp_path, "train", shard_size=1) as writer:
        writer.append(state, labels, {"episode": 0, "step": 0})
    dataset = LabelDataset(tmp_path, "train")
    batch = dataset.sample(3, np.random.default_rng(1), hierarchy=True, local_budget=24, region_budget=8)
    assert batch.state.node_features.shape == (3, 32, 4)
    assert torch.equal(batch.future_gain_mask, batch.state.candidate_mask)
    assert torch.isfinite(batch.future_gain[batch.future_gain_mask]).all()


def test_exact_global_batch_schedule_for_three_ranks():
    schedules = [BatchSchedule(128, 3, rank, 32) for rank in range(3)]
    assert [item.local_samples for item in schedules] == [43, 43, 42]
    assert sum(item.local_samples for item in schedules) == 128
    assert all(len(item.chunk_sizes) == 2 for item in schedules)


def test_fixed_global_batch_across_dynamic_world_sizes():
    expected_accumulation = {1: 4, 2: 2, 4: 1}
    for world_size, expected in expected_accumulation.items():
        schedules = [BatchSchedule(128, world_size, rank, 32) for rank in range(world_size)]
        assert sum(item.local_samples for item in schedules) == 128
        assert max(len(item.chunk_sizes) for item in schedules) == expected


def test_calibration_loader_preserves_variance_decomposition(tmp_path: Path):
    import json

    from ac_pbgrl.learning.calibration import load_variance_temperatures

    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps({"region_temperature": 4.0, "action_temperature": 2.0}),
        encoding="utf-8",
    )
    region, action = load_variance_temperatures(calibration)
    assert region == action == 2.0


def test_gru_control_uses_trained_log_variance_parameterization():
    memory = GRUPotentialMemory(hidden_dim=4, ttl_steps=8, seed=1)
    with torch.no_grad():
        memory.output.weight.zero_()
        memory.output.bias[:] = torch.tensor([0.0, np.log(2.0)])
    _, variance = memory.update_many(
        np.asarray([11]), np.asarray([0.5]), np.asarray([0.2]), step=0
    )
    assert np.isclose(variance[0], 2.0)


def test_replay_survives_reopen_and_sac_updates(tmp_path: Path):
    config = smoke_config()
    environment = SyntheticGraphExplorationEnv(32, 8, seed=4)
    state, critic = environment.reset()
    result = environment.step(0)
    transition = TransitionBatch(
        state=state,
        action=torch.tensor([0]),
        reward=torch.tensor([result.reward]),
        done=torch.tensor([float(result.done)]),
        next_state=result.state,
        critic_state=critic,
        critic_next_state=result.critic_state,
        future_gain=torch.zeros(1, 8),
        future_gain_mask=state.candidate_mask,
    )
    replay = PersistentReplayBuffer(tmp_path / "replay", 16)
    replay.add(transition)
    reopened = PersistentReplayBuffer(tmp_path / "replay", 16)
    assert len(reopened) == 1
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    learner = DiscreteSACLearner(config, context)
    schedule = BatchSchedule(8, 1, 0, 4)
    chunks = [reopened.sample(size) for size in schedule.chunk_sizes]
    metrics = learner.update_chunks(chunks, schedule)
    assert np.isfinite(metrics["loss/policy_sac"])
    assert np.isfinite(metrics["loss/region_potential"])


def test_sac_twin_critics_are_independently_initialized():
    config = smoke_config()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    learner = DiscreteSACLearner(config, context)

    parameter_pairs = zip(learner.q1_raw.parameters(), learner.q2_raw.parameters())
    assert any(not torch.equal(q1, q2) for q1, q2 in parameter_pairs)
    assert all(
        torch.equal(online, target)
        for online, target in zip(learner.q1_raw.parameters(), learner.target_q1.parameters())
    )
    assert all(
        torch.equal(online, target)
        for online, target in zip(learner.q2_raw.parameters(), learner.target_q2.parameters())
    )

    invalid_checkpoint = learner.state_dict()
    invalid_checkpoint["q2"] = copy.deepcopy(invalid_checkpoint["q1"])
    with pytest.raises(ValueError, match="bit-identical q1/q2"):
        learner.load_state_dict(invalid_checkpoint)


def test_real_map_actor_critic_action_coordinates_align():
    from ac_pbgrl.envs.ariadne.adapter import AriadneExplorationEnv

    maps = Path(__file__).resolve().parents[1] / "maps"
    environment = AriadneExplorationEnv(
        maps_dir=maps,
        node_padding=360,
        critic_node_padding=512,
        candidate_padding=25,
        max_episode_steps=2,
        hierarchy=True,
        local_budget=192,
        region_budget=32,
        seed=0,
    )
    state, critic = environment.reset(episode=0, map_path=sorted(maps.glob("*.png"))[0])
    mask = state.candidate_mask[0]
    actor_xy = state.node_xy[0, state.candidate_indices[0, mask]]
    critic_xy = critic.node_xy[0, critic.candidate_indices[0, mask]]
    assert torch.allclose(actor_xy, critic_xy)
    action = int(torch.nonzero(mask, as_tuple=False)[0])
    result = environment.step(action)
    assert np.isfinite(result.reward)


@pytest.mark.integration
def test_onnx_export_matches_torch(tmp_path: Path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from ac_pbgrl.export import compare_onnx, export_onnx

    config = smoke_config()
    policy = ACPolicyNetwork(4, 4, 32, 4, 2, use_potential=True, use_diffusion=True)
    checkpoint = tmp_path / "actor.pt"
    torch.save({"actor": policy.state_dict()}, checkpoint)
    output = tmp_path / "policy.onnx"
    export_onnx(config, checkpoint, output)
    assert compare_onnx(config, checkpoint, output) < 1.0e-4
