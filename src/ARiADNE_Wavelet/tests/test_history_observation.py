import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np

from ARiADNE_Wavelet.agent import Agent
from ARiADNE_Wavelet import env as env_module
from ARiADNE_Wavelet.env import Env
from ARiADNE_Wavelet.model import PolicyNet
from ARiADNE_Wavelet.parameter import EMBEDDING_DIM, HISTORY_INPUT_DIM, HISTORY_LEN, NODE_INPUT_DIM


class HistoryObservationTests(unittest.TestCase):
    @staticmethod
    def _create_test_map(map_dir: Path) -> Path:
        map_dir.mkdir(parents=True, exist_ok=True)
        grid = np.full((32, 32), 255, dtype=np.uint8)
        grid[16, 16] = 208
        map_path = map_dir / "test_map.png"
        imageio.imwrite(map_path, grid)
        return map_path

    def test_agent_observation_includes_history_tensor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_root = Path(tmpdir) / "maps"
            self._create_test_map(map_root)
            with patch.object(env_module, "MAPS_DIR", map_root):
                env = Env(0, plot=False)
                model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, enable_wavelet_history=False)
                agent = Agent(model, device="cpu", plot=False)

                agent.reset_history()
                agent.update_planning_state(env.belief_info, env.robot_location)
                observation = agent.get_observation()

                self.assertEqual(len(observation), 7)
                history_inputs = observation[6]
                self.assertEqual(tuple(history_inputs.shape), (1, HISTORY_LEN, HISTORY_INPUT_DIM))

    def test_history_buffer_rolls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_root = Path(tmpdir) / "maps"
            self._create_test_map(map_root)
            with patch.object(env_module, "MAPS_DIR", map_root):
                env = Env(0, plot=False)
                model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, enable_wavelet_history=False)
                agent = Agent(model, device="cpu", plot=False)

                agent.reset_history()
                for _ in range(HISTORY_LEN + 3):
                    agent.update_planning_state(env.belief_info, env.robot_location)
                self.assertEqual(len(agent.history_buffer), HISTORY_LEN)


if __name__ == "__main__":
    unittest.main()
