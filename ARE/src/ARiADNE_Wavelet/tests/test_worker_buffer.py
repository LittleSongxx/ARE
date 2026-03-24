import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import imageio.v2 as imageio
import torch
import numpy as np

from ARiADNE_Wavelet import parameter
from ARiADNE_Wavelet import env as env_module
from ARiADNE_Wavelet.model import PolicyNet
from ARiADNE_Wavelet.parameter import EMBEDDING_DIM, NODE_INPUT_DIM, RuntimeConfig
from ARiADNE_Wavelet.worker import Worker


class WorkerBufferTests(unittest.TestCase):
    @staticmethod
    def _create_test_map(map_dir: Path) -> Path:
        map_dir.mkdir(parents=True, exist_ok=True)
        grid = np.full((32, 32), 255, dtype=np.uint8)
        grid[16, 16] = 208
        map_path = map_dir / "test_map.png"
        imageio.imwrite(map_path, grid)
        return map_path

    def test_worker_produces_18_channel_replay(self):
        torch.manual_seed(7)
        np.random.seed(7)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_root = root / "maps"
            self._create_test_map(map_root)
            with patch.object(parameter, "RESULT_ROOT", root / "result"), patch.object(env_module, "MAPS_DIR", map_root):
                runtime_config = RuntimeConfig(
                    max_episodes=1,
                    num_meta_agent=1,
                    max_episode_step=1,
                    minimum_buffer_size=1,
                    batch_size=1,
                    replay_size=8,
                    save_img_gap=1000,
                    summary_window=1,
                    train_updates_per_iter=1,
                    run_session="test_worker_buffer",
                )
                model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, enable_wavelet_history=False)
                worker = Worker(0, model, 1, runtime_config=runtime_config, device="cpu", save_image=False)
                worker.run_episode()

                self.assertEqual(len(worker.episode_buffer), 18)
                self.assertGreaterEqual(len(worker.episode_buffer[0]), 1)
                self.assertEqual(len(worker.episode_buffer[0]), len(worker.episode_buffer[15]))
                self.assertEqual(len(worker.episode_buffer[9]), len(worker.episode_buffer[16]))
                self.assertEqual(len(worker.episode_buffer[0]), len(worker.episode_buffer[17]))


if __name__ == "__main__":
    unittest.main()
