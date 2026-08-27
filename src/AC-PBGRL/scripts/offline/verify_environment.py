#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys


PINS = {
    "numpy": "1.24.3",
    "scipy": "1.10.1",
    "scikit-image": "0.21.0",
    "matplotlib": "3.7.5",
    "tensorboard": "2.14.0",
    "PyYAML": "6.0.3",
    "ray": "2.10.0",
    "h5py": "3.11.0",
    "pandas": "2.0.3",
    "seaborn": "0.13.2",
    "scikit-learn": "1.3.2",
    "onnx": "1.16.2",
    "onnxruntime": "1.16.3",
    "pytest": "8.3.5",
    "pytest-cov": "5.0.0",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args(argv)
    report = {"python": sys.version.split()[0], "packages": {}, "errors": []}
    if not sys.version.startswith("3.8."):
        report["errors"].append("server/ROS lock requires Python 3.8")
    for package, expected in PINS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        report["packages"][package] = actual
        if actual != expected:
            report["errors"].append(f"{package}: expected {expected}, found {actual}")
    if args.server:
        try:
            import torch

            report["packages"]["torch"] = torch.__version__
            report["cuda"] = torch.version.cuda
            report["cuda_available"] = torch.cuda.is_available()
            if not str(torch.__version__).startswith("2.4.1+cu121"):
                report["errors"].append(
                    f"torch: expected 2.4.1+cu121, found {torch.__version__}"
                )
            if str(torch.version.cuda) != "12.1":
                report["errors"].append(f"torch CUDA: expected 12.1, found {torch.version.cuda}")
            if not torch.cuda.is_available():
                report["errors"].append("CUDA is not available to PyTorch")
        except ImportError:
            report["errors"].append("torch is not installed")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
