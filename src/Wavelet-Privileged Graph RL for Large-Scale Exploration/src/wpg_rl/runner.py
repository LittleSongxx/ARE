from __future__ import annotations

import os

import ray

from .parameter import EMBEDDING_DIM, NODE_INPUT_DIM, RuntimeConfig, apply_runtime_config
from .runtime_utils import configure_matplotlib_cache, configure_worker_process_threads, resolve_worker_num_threads


class Runner:
    def __init__(self, meta_agent_id, runtime_config: RuntimeConfig):
        self.meta_agent_id = meta_agent_id
        self.runtime_config = apply_runtime_config(runtime_config)
        self.worker_num_threads = resolve_worker_num_threads(self.runtime_config)
        configure_worker_process_threads(self.worker_num_threads)
        configure_matplotlib_cache(f"runner-{self.meta_agent_id}-pid-{os.getpid()}")

        import torch

        from .model import PolicyNet
        from .worker import Worker

        torch.set_num_threads(self.worker_num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(self.worker_num_threads)
            except RuntimeError:
                pass

        self.Worker = Worker
        self.device = torch.device("cpu")
        self.network = PolicyNet(
            NODE_INPUT_DIM,
            EMBEDDING_DIM,
            use_lf_attention_hf_residual=self.runtime_config.use_lf_attention_hf_residual,
            use_privileged_wavelet_distillation=self.runtime_config.use_privileged_wavelet_distillation,
            wavelet_scales=self.runtime_config.wavelet_scales,
            wavelet_fuse_dim=self.runtime_config.wavelet_fuse_dim,
            wavelet_lf_qk=self.runtime_config.wavelet_lf_qk,
        ).to(self.device)

        print(
            f"runner_init metaAgent={self.meta_agent_id} "
            f"device={self.device.type} "
            f"worker_num_threads={self.worker_num_threads}"
        )

    def get_weights(self):
        return self.network.state_dict()

    def set_policy_net_weights(self, weights):
        self.network.load_state_dict(weights)

    def do_job(self, episode_number):
        save_gap = max(int(self.runtime_config.save_img_gap), 1)
        save_img = episode_number % save_gap == 0
        worker = self.Worker(
            self.meta_agent_id,
            self.network,
            episode_number,
            device=self.device,
            save_image=save_img,
        )
        worker.run_episode()
        return worker.episode_buffer, worker.perf_metrics

    def job(self, weights_set, episode_number):
        print(f"starting episode {episode_number} on metaAgent {self.meta_agent_id}")
        self.set_policy_net_weights(weights_set[0])
        job_results, metrics = self.do_job(episode_number)
        info = {"id": self.meta_agent_id, "episode_number": episode_number}
        return job_results, metrics, info
