from __future__ import annotations

import ray
import torch

from .model import PolicyNet
from .parameter import EMBEDDING_DIM, NODE_INPUT_DIM, RuntimeConfig
from .runtime_utils import configure_worker_process_threads, resolve_worker_num_threads
from .worker import Worker


def _cuda_runtime_status() -> tuple[bool, str | None]:
    if not torch.cuda.is_available():
        return False, "No CUDA GPUs are available"
    try:
        probe = torch.zeros(1, device="cuda")
        probe = probe + 1
        probe.item()
        torch.cuda.synchronize()
    except (AssertionError, RuntimeError) as exc:
        return False, str(exc)
    return True, None


class Runner:
    def __init__(self, meta_agent_id, runtime_config: RuntimeConfig):
        self.meta_agent_id = meta_agent_id
        self.runtime_config = runtime_config
        self.worker_num_threads = resolve_worker_num_threads(runtime_config)
        configure_worker_process_threads(self.worker_num_threads)

        torch.set_num_threads(self.worker_num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(self.worker_num_threads)
            except RuntimeError:
                pass

        self.device = torch.device("cpu")

        if runtime_config.use_gpu:
            cuda_ok, reason = _cuda_runtime_status()
            if not cuda_ok:
                print(f"Runner[{meta_agent_id}] falling back to CPU: {reason}")
            else:
                assigned_gpu_ids = ray.get_gpu_ids()
                if assigned_gpu_ids:
                    torch.cuda.set_device(0)
                    self.device = torch.device("cuda:0")
                else:
                    print(f"Runner[{meta_agent_id}] falling back to CPU: Ray assigned no GPU resources")

        self.network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
        try:
            self.network.to(self.device)
        except (AssertionError, RuntimeError) as exc:
            if self.device.type != "cuda":
                raise
            print(f"Runner[{meta_agent_id}] falling back to CPU: {exc}")
            self.device = torch.device("cpu")
            self.network.to(self.device)

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
        worker = Worker(
            self.meta_agent_id,
            self.network,
            episode_number,
            runtime_config=self.runtime_config,
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
