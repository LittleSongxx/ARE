import torch

from env import Env
from agent import Agent
from utils import *
from model import PolicyNet
from ground_truth_node_manager import GroundTruthNodeManager


class Worker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.gifs_path = str(get_gifs_path())
        os.makedirs(self.gifs_path, exist_ok=True)

        self.env = Env(global_step, plot=self.save_image)
        self.robot = Agent(policy_net, self.device, self.save_image)

        self.ground_truth_node_manager = GroundTruthNodeManager(self.robot.node_manager, self.env.ground_truth_info,
                                                                device=self.device, plot=self.save_image)
        self.episode_return = 0.0
        self.episode_steps = 0

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(27):
            self.episode_buffer.append([])

    @staticmethod
    def _buffer_tensor(tensor):
        return tensor.detach().to('cpu')

    def run_episode(self):
        done = False
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()
        ground_truth_observation = self.ground_truth_node_manager.get_ground_truth_observation(self.env.robot_location)

        if self.save_image:
            self.robot.plot_env()
            self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
            self.env.plot_env(0)

        for i in range(MAX_EPISODE_STEP):
            self.save_observation(observation, ground_truth_observation)

            next_location, action_index = self.robot.select_next_waypoint(observation)
            self.save_action(action_index)

            node = self.robot.node_manager.nodes_dict.find((self.robot.location[0], self.robot.location[1]))
            check = np.array(list(node.data.neighbor_set)).reshape(-1, 2)
            assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, print(next_location, self.robot.location, node.data.neighbor_set)
            assert next_location[0] != self.robot.location[0] or next_location[1] != self.robot.location[1]

            reward = self.env.step(next_location)

            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            if self.robot.utility.sum() == 0:
                done = True
                reward += 20
            self.episode_return += float(reward)
            self.episode_steps = i + 1
            self.save_reward_done(reward, done)

            observation = self.robot.get_observation()
            ground_truth_observation = self.ground_truth_node_manager.get_ground_truth_observation(
                self.env.robot_location)
            self.save_next_observations(observation, ground_truth_observation)

            if self.save_image:
                self.robot.plot_env()
                self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
                self.env.plot_env(i+1)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['episode_return'] = self.episode_return
        self.perf_metrics['episode_steps'] = self.episode_steps

        # save gif
        if self.save_image:
            # 保存到以训练轮数为名的子目录
            episode_gifs_path = os.path.join(self.gifs_path, f'episode_{self.global_step}')
            if not os.path.exists(episode_gifs_path):
                os.makedirs(episode_gifs_path)
            make_gif(episode_gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def save_observation(self, observation, ground_truth_observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[0] += self._buffer_tensor(node_inputs)
        self.episode_buffer[1] += self._buffer_tensor(node_padding_mask.bool())
        self.episode_buffer[2] += self._buffer_tensor(edge_mask.bool())
        self.episode_buffer[3] += self._buffer_tensor(current_index)
        self.episode_buffer[4] += self._buffer_tensor(current_edge)
        self.episode_buffer[5] += self._buffer_tensor(edge_padding_mask.bool())

        critic_node_inputs, critic_node_padding_mask, critic_edge_mask, critic_current_index, critic_current_edge, critic_edge_padding_mask = ground_truth_observation
        self.episode_buffer[15] += self._buffer_tensor(critic_node_inputs)
        self.episode_buffer[16] += self._buffer_tensor(critic_node_padding_mask.bool())
        self.episode_buffer[17] += self._buffer_tensor(critic_edge_mask.bool())
        self.episode_buffer[18] += self._buffer_tensor(critic_current_index)
        self.episode_buffer[19] += self._buffer_tensor(critic_current_edge)
        self.episode_buffer[20] += self._buffer_tensor(critic_edge_padding_mask.bool())

        assert torch.all(current_edge == critic_current_edge), print(current_edge, critic_current_edge, current_index, critic_current_index)
        assert torch.all(node_inputs[0, current_index.item(), :2] == critic_node_inputs[0, critic_current_index.item(), :2]), print(node_inputs[0, current_index.item()], critic_node_inputs[0, critic_current_index.item()])
        assert torch.all(torch.gather(node_inputs, 1, current_edge.repeat(1, 1, 2)) == torch.gather(critic_node_inputs, 1, critic_current_edge.repeat(1, 1, 2)))

    def save_action(self, action_index):
        self.episode_buffer[6] += self._buffer_tensor(action_index.reshape(1, 1, 1))

    def save_reward_done(self, reward, done):
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        done_tensor = torch.tensor([int(done)], dtype=torch.int64, device=self.device).reshape(1, 1, 1)
        self.episode_buffer[7] += self._buffer_tensor(reward_tensor)
        self.episode_buffer[8] += self._buffer_tensor(done_tensor)

    def save_next_observations(self, observation, ground_truth_observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[9] += self._buffer_tensor(node_inputs)
        self.episode_buffer[10] += self._buffer_tensor(node_padding_mask.bool())
        self.episode_buffer[11] += self._buffer_tensor(edge_mask.bool())
        self.episode_buffer[12] += self._buffer_tensor(current_index)
        self.episode_buffer[13] += self._buffer_tensor(current_edge)
        self.episode_buffer[14] += self._buffer_tensor(edge_padding_mask.bool())

        critic_node_inputs, critic_node_padding_mask, critic_edge_mask, critic_current_index, critic_current_edge, critic_edge_padding_mask = ground_truth_observation
        self.episode_buffer[21] += self._buffer_tensor(critic_node_inputs)
        self.episode_buffer[22] += self._buffer_tensor(critic_node_padding_mask.bool())
        self.episode_buffer[23] += self._buffer_tensor(critic_edge_mask.bool())
        self.episode_buffer[24] += self._buffer_tensor(critic_current_index)
        self.episode_buffer[25] += self._buffer_tensor(critic_current_edge)
        self.episode_buffer[26] += self._buffer_tensor(critic_edge_padding_mask.bool())

if __name__ == "__main__":
    torch.manual_seed(4777)
    np.random.seed(4777)
    model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    # checkpoint = torch.load(model_path + '/checkpoint.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['policy_model'])
    worker = Worker(0, model, 77, save_image=False)
    worker.run_episode()
