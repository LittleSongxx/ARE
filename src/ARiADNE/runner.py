from __future__ import annotations

import torch

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT.parent))

    from ARiADNE.model import PolicyNet
    from ARiADNE.parameter import EMBEDDING_DIM, RuntimeConfig, get_node_input_dim
    from ARiADNE.runtime_utils import configure_worker_process_threads
    from ARiADNE.worker import Worker
else:
    from .model import PolicyNet
    from .parameter import EMBEDDING_DIM, RuntimeConfig, get_node_input_dim
    from .runtime_utils import configure_worker_process_threads
    from .worker import Worker


class Runner:
    def __init__(self, meta_agent_id, runtime_config: RuntimeConfig):
        self.meta_agent_id = meta_agent_id
        self.runtime_config = runtime_config
        if runtime_config.worker_num_threads is not None:
            configure_worker_process_threads(runtime_config.worker_num_threads)
            torch.set_num_threads(max(int(runtime_config.worker_num_threads), 1))

        self.device = torch.device("cuda") if runtime_config.use_gpu else torch.device("cpu")
        self.local_network = PolicyNet(get_node_input_dim(runtime_config), EMBEDDING_DIM)
        self.local_network.to(self.device)

    def get_weights(self):
        return self.local_network.state_dict()

    def set_policy_net_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number):
        save_img = episode_number % self.runtime_config.save_img_gap == 0
        worker = Worker(
            self.meta_agent_id,
            self.local_network,
            episode_number,
            runtime_config=self.runtime_config,
            device=self.device,
            save_image=save_img,
        )
        worker.run_episode()
        return worker.episode_buffer, worker.perf_metrics

    def job(self, weights_set, episode_number):
        self.set_policy_net_weights(weights_set[0])
        job_results, metrics = self.do_job(episode_number)
        info = {"id": self.meta_agent_id, "episode_number": episode_number}
        return job_results, metrics, info
