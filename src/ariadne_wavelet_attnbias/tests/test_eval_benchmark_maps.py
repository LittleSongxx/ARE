import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ariadne_wavelet_attnbias.evaluation import resolve_benchmark_eval_maps
from ariadne_wavelet_attnbias.parameter import RuntimeConfig


class EvalBenchmarkMapTests(unittest.TestCase):
    def _touch(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("map")

    def test_named_benchmark_maps_are_resolved_in_declared_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._touch(root / "hard_a.png")
            self._touch(root / "hard_b.png")
            self._touch(root / "easy_a.png")

            config = RuntimeConfig(eval_benchmark_maps=("hard_b.png", "hard_a.png"))
            with patch("ariadne_wavelet_attnbias.evaluation.MAPS_DIR", root):
                benchmark_maps = resolve_benchmark_eval_maps(config)

            self.assertEqual([path.name for path in benchmark_maps], ["hard_b.png", "hard_a.png"])

    def test_empty_benchmark_list_falls_back_to_stable_first_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._touch(root / "b.png")
            self._touch(root / "a.png")

            config = RuntimeConfig(eval_benchmark_maps=tuple())
            with patch("ariadne_wavelet_attnbias.evaluation.MAPS_DIR", root):
                benchmark_maps = resolve_benchmark_eval_maps(config)

            self.assertEqual([path.name for path in benchmark_maps], ["a.png"])


if __name__ == "__main__":
    unittest.main()
