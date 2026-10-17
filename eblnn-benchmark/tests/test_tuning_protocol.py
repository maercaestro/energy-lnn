from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from eblnn.src.data_real import RealDataPipeline as TuningPipeline
from eblnn.src.model import create_model as create_tuning_model
from eblnn.src.sampler import build_sampler
from eblnn.src.train import GenerativeTrainer
from src.data import INPUT_COLS, TARGET_COLS, RealDataPipeline
from src.models import create_model
from src.trainers import create_trainer
from scripts import run_benchmark


class TuningProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(ROOT / "config" / "benchmark_config.yaml") as handle:
            cls.config = yaml.safe_load(handle)
        with open(ROOT.parent / "eblnn" / "ebm_tuning" / "base_config.yaml") as handle:
            cls.tuning = yaml.safe_load(handle)

    def test_frozen_configuration_and_protocol(self):
        cfg = self.config
        selected = {
            "alpha": 1.0, "l2_reg": 0.1, "n_steps": 20,
            "step_size": 0.01, "noise_scale": 0.005, "buffer_prob": 0.75,
        }
        flattened = {**cfg["eblnn"], **cfg["eblnn"]["langevin"]}
        for key, value in selected.items():
            self.assertEqual(flattened[key], value, key)
        for section in ("data", "training", "eblnn"):
            for key, value in cfg[section].items():
                if key in self.tuning and key not in selected and key not in ("real_csv", "edge_csv"):
                    self.assertEqual(value, self.tuning[key], key)
        for key in ("hidden_size", "ebm_hidden_dims"):
            self.assertEqual(cfg["models"]["eblnn"][key], self.tuning[key])
        self.assertTrue(cfg["models"]["eblnn"]["mixed_memory"])
        self.assertEqual(cfg["run_name"], "ebm-tuned-f6ozllp7")

    def write_fixture_data(self, directory):
        rng = np.random.default_rng(42)
        columns = INPUT_COLS + TARGET_COLS
        real = pd.DataFrame(rng.normal(size=(600, len(columns))), columns=columns)
        edge = pd.DataFrame(rng.normal(size=(600, len(columns))), columns=columns)
        edge["scenario"] = np.repeat(["a", "b"], 300)
        real_path, edge_path = Path(directory) / "real.csv", Path(directory) / "edge.csv"
        real.to_csv(real_path, index=False)
        edge.to_csv(edge_path, index=False)
        return {
            **self.config["data"],
            "real_csv": str(real_path), "edge_csv": str(edge_path),
        }

    def test_data_pipeline_matches_tuning(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = {**self.write_fixture_data(directory), "seed": 42}
            with contextlib.redirect_stdout(io.StringIO()):
                tuning = TuningPipeline(**kwargs).build()
                benchmark = RealDataPipeline(**kwargs).build()
            for key, value in tuning.data_summary.items():
                self.assertEqual(benchmark.data_summary[key], value)
            for split in ("train", "val", "test"):
                expected = getattr(tuning, f"{split}_loader").dataset.tensors
                actual = getattr(benchmark, f"{split}_loader").dataset.tensors
                for left, right in zip(expected, actual):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            for name in ("input_scaler", "target_scaler"):
                for attr in ("mean_", "scale_"):
                    np.testing.assert_array_equal(
                        getattr(getattr(tuning, name), attr),
                        getattr(getattr(benchmark, name), attr),
                    )

    def test_smoke_isolation_and_saved_diagnostics(self):
        class WithoutTestLoader(RealDataPipeline):
            def build(self):
                super().build()
                self.test_loader = None
                return self

        with tempfile.TemporaryDirectory() as directory:
            config = {**self.config, "data": self.write_fixture_data(directory)}
            config_path = Path(directory) / "smoke_config.yaml"
            with open(config_path, "w") as handle:
                yaml.safe_dump(config, handle)
            output = Path(directory) / "outputs"
            argv = [
                "run_benchmark", "--config", str(config_path), "--smoke-test",
                "--models", "eblnn", "--output_dir", str(output),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(run_benchmark, "RealDataPipeline", WithoutTestLoader),
                mock.patch.object(run_benchmark, "_WANDB_AVAILABLE", False),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run_benchmark.main()
            self.assertFalse((output / "eblnn").exists())
            smoke = output / "smoke"
            manifest = json.loads((smoke / "run_manifest.json").read_text())
            self.assertEqual(manifest["seeds"], [42])
            self.assertTrue(manifest["smoke_test"])
            self.assertTrue(manifest["run_name"].endswith("-smoke"))
            result_dir = smoke / "eblnn" / "seed_42"
            result = json.loads((result_dir / "results.json").read_text())
            self.assertEqual(result["evaluation_split"], "validation")
            self.assertTrue(result["smoke_test"])
            self.assertEqual(result["training"]["epochs_run"], 2)
            self.assertLessEqual(result["inference"]["n_samples"], 16)
            self.assertTrue((result_dir / "models" / "best_model.pth").exists())
            with np.load(result_dir / "history.npz") as history:
                for key in ("train_phys", "val_phys", "train_cd", "val_cd", "cd_gap"):
                    self.assertEqual(len(history[key]), 2)

    def test_training_and_disk_checkpoint_match_tuning(self):
        cfg = self.config
        flat = {
            **self.tuning, **cfg["eblnn"], **cfg["eblnn"]["langevin"],
            "epochs": 2, "heartbeat": 0,
        }
        generator = torch.Generator().manual_seed(7)
        dataset = TensorDataset(
            torch.randn(2, 30, 5, generator=generator),
            torch.randn(2, 30, 2, generator=generator),
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(42)
            tuning_model = create_tuning_model(device="cpu")
            tuning = GenerativeTrainer(
                tuning_model,
                build_sampler(tuning_model, **{
                    key: flat[key] for key in ("n_steps", "step_size", "noise_scale", "clip_x")
                }),
                flat, device="cpu", use_wandb=False,
            )
            tuning.train(loader, loader, str(Path(directory) / "tuning"))

            torch.manual_seed(42)
            model = create_model("eblnn", 5, 2, cfg["models"]["eblnn"], "cpu")
            benchmark = create_trainer(
                "eblnn", model, flat, "cpu", eblnn_cfg=cfg["eblnn"],
                seq_len=30, input_size=5,
            )
            benchmark.train(loader, loader, str(Path(directory) / "benchmark"), verbose=False)
            for key in benchmark.history:
                if key != "val_loss":
                    np.testing.assert_allclose(
                        benchmark.history[key], tuning.history[key], rtol=0, atol=0,
                        err_msg=key,
                    )
            self.assertEqual(benchmark.history["val_loss"], tuning.history["val_phys"])
            self.assertEqual(benchmark.best_epoch, tuning.best_epoch)
            self.assertEqual(benchmark.best_val_loss, tuning.best_val_loss)
            tuning_best = torch.load(
                Path(directory) / "tuning" / "best_model.pth", weights_only=True,
            )
            benchmark.best_state_dict = None
            benchmark.load_best_model()
            for key, value in tuning_best.items():
                torch.testing.assert_close(model.core.state_dict()[key], value, rtol=0, atol=0)
            truth, prediction = benchmark.predict(loader)
            self.assertEqual(truth.shape, (60, 2))
            self.assertEqual(prediction.shape, truth.shape)
            self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
