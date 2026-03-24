import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ARiADNE_Wavelet import parameter
from ARiADNE_Wavelet.parameter import (
    RuntimeConfig,
    ensure_output_dirs,
    get_checkpoint_path,
    get_gifs_path,
    get_model_path,
    get_result_eval_path,
    get_result_eval_gifs_path,
    get_train_path,
)


class ArtifactLayoutTests(unittest.TestCase):
    def test_layout_under_result_run_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.object(parameter, "RESULT_ROOT", root / "result"):
                config = RuntimeConfig(run_session="2026_0312_0101")
                ensure_output_dirs(config)

                self.assertTrue(get_model_path(config).is_dir())
                self.assertTrue(get_train_path(config).is_dir())
                self.assertTrue(get_gifs_path(config).is_dir())
                self.assertTrue(get_result_eval_path(config).is_dir())
                self.assertTrue(get_result_eval_gifs_path(config).is_dir())
                self.assertEqual(
                    get_checkpoint_path(config),
                    root / "result" / "2026_0312_0101" / "train" / "model" / "checkpoint.pth",
                )


if __name__ == "__main__":
    unittest.main()
