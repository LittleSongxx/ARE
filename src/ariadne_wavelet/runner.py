import torch

from .model import PolicyNet
from .parameter import EMBEDDING_DIM, NODE_INPUT_DIM, RuntimeConfig
from .worker import Worker


class Runner:
    def __init__(self, meta_agent_id, runtime_config: RuntimeConfig):
        self.meta_agent_id = meta_agent_id
        self.runtime_config = runtime_config
        self.device = torch.device("cuda") if runtime_config.use_gpu else torch.device("cpu")
        self.network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
        self.network.to(self.device)

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
        self.set_policy_net_weights(weights_set[0])
        job_results, metrics = self.do_job(episode_number)
        info = {"id": self.meta_agent_id, "episode_number": episode_number}
        return job_results, metrics, info
