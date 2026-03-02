import torch
import ray
from model import PolicyNet
from worker import Worker
from parameter import *


class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.assigned_gpu_ids = [int(gpu_id) for gpu_id in ray.get_gpu_ids()]
        if USE_GPU and torch.cuda.is_available():
            if self.assigned_gpu_ids:
                # Ray scopes visible devices for the worker, so the first assigned
                # GPU is exposed locally as cuda:0.
                torch.cuda.set_device(0)
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        self.network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
        self.network.to(self.device)

    def get_weights(self):
        return self.network.state_dict()

    def set_policy_net_weights(self, weights):
        self.network.load_state_dict(weights)

    def do_job(self, episode_number):
        save_img = True if episode_number % SAVE_IMG_GAP == 0 else False
        # save_img = True
        worker = Worker(self.meta_agent_id, self.network, episode_number, device=self.device, save_image=save_img)
        worker.run_episode()

        job_results = worker.episode_buffer
        perf_metrics = worker.perf_metrics
        return job_results, perf_metrics

    def job(self, weights_set, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_policy_net_weights(weights_set[0])

        job_results, metrics = self.do_job(episode_number)

        info = {"id": self.meta_agent_id, "episode_number": episode_number}

        return job_results, metrics, info


# Each worker gets a fraction of GPU resources
# Single GPU: num_gpus=0, Multi-GPU (4xA40, 64 agents): num_gpus=4/64=0.0625
@ray.remote(num_cpus=1, num_gpus=NUM_GPU / NUM_META_AGENT if USE_GPU and NUM_GPU > 0 else 0)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):
        super().__init__(meta_agent_id)


if __name__ == '__main__':
    if USE_GPU and NUM_GPU > 0:
        ray.init(num_gpus=NUM_GPU)
    else:
        ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.do_job.remote(1)
    out = ray.get(job_id)
    print(out[1])
