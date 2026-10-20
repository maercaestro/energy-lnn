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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import compare_results, run_benchmark
from src.data_forecast import FORECAST_FEATURES, SENSOR_COLUMNS, ForecastDataPipeline
from src.models import create_model


class ForecastProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        n = 1200
        dates = np.arange(n) + 1
        dates[400:] += 20
        dates[1000:] += 50
        values = np.arange(n, dtype=float)
        self.frame = pd.DataFrame({
            column: 2 * index + values * (index + 1) / 100
            for index, column in enumerate(SENSOR_COLUMNS.values())
        })
        self.frame["Date"] = dates
        self.path = self.root / "cleaned.csv"
        self.frame.to_csv(self.path, index=False)

    def pipeline(self, **overrides):
        kwargs = {
            "real_csv": str(self.path), "seq_len": 30, "stride": 30,
            "batch_size": 8, "val_size": .1, "test_size": .1,
        }
        return ForecastDataPipeline(**{**kwargs, **overrides}).build()

    def test_future_alignment_gaps_and_disjoint_splits(self):
        for horizon in (1, 3):
            pipeline = self.pipeline(horizon=horizon)
            previous_last = -1
            source = self.frame.set_index("Date")
            for split in ("train", "val", "test"):
                inputs, targets = pipeline.window_dates[split]
                self.assertTrue((targets - inputs == horizon).all())
                self.assertTrue((np.diff(inputs, axis=1) == 1).all())
                self.assertTrue((np.diff(targets, axis=1) == 1).all())
                self.assertGreater(inputs.min(), previous_last)
                previous_last = targets.max()
                x, y = getattr(pipeline, f"{split}_loader").dataset.tensors
                raw_x = pipeline.input_scaler.inverse_transform(x.numpy().reshape(-1, 9))
                raw_y = pipeline.target_scaler.inverse_transform(y.numpy().reshape(-1, 2))
                np.testing.assert_allclose(
                    raw_x, source.loc[inputs.ravel(), list(SENSOR_COLUMNS.values())],
                    rtol=1e-6, atol=1e-5,
                )
                np.testing.assert_allclose(
                    raw_y, source.loc[targets.ravel(), ["OutletT", "ExcessO2"]],
                    rtol=1e-6, atol=1e-5,
                )

    def test_training_scalers_and_splits_do_not_depend_on_future_or_seed(self):
        first = self.pipeline(seed=42)
        second = self.pipeline(seed=123)
        for split in ("train", "val", "test"):
            for a, b in zip(first.window_dates[split], second.window_dates[split]):
                np.testing.assert_array_equal(a, b)
        self.frame.loc[960:, list(SENSOR_COLUMNS.values())] += 10000
        self.frame.to_csv(self.path, index=False)
        changed = self.pipeline()
        for name in ("input_scaler", "target_scaler"):
            for stat in ("mean_", "scale_"):
                np.testing.assert_array_equal(
                    getattr(getattr(first, name), stat),
                    getattr(getattr(changed, name), stat),
                )
        for a, b in zip(first.train_loader.dataset.tensors, changed.train_loader.dataset.tensors):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_reject_invalid_source_and_protocol_parameters(self):
        for kwargs in (
            {"horizon": 0}, {"horizon": 1.5}, {"edge_csv": "snapshot.csv"},
            {"val_size": .6, "test_size": .5}, {"stride": 0}, {"seq_len": 10000},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.pipeline(**kwargs)
        self.frame.loc[1, "Date"] = self.frame.loc[0, "Date"]
        self.frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.pipeline()

    def test_default_configuration_is_forecasting(self):
        with mock.patch.object(sys, "argv", ["benchmark"]):
            args = run_benchmark.parse_args()
        self.assertEqual(Path(args.config).name, "forecast_config.yaml")

    def test_persistence_uses_current_measurements_and_reloads(self):
        pipeline = self.pipeline()
        model = create_model(
            "persistence", 9, 2, {}, "cpu",
            input_scaler=pipeline.input_scaler, target_scaler=pipeline.target_scaler,
            feature_names=FORECAST_FEATURES,
        )
        x, y = next(iter(pipeline.val_loader))
        prediction = model(x)[0]
        observed = pipeline.input_scaler.inverse_transform(x.reshape(-1, 9).numpy())
        predicted = pipeline.target_scaler.inverse_transform(prediction.reshape(-1, 2).numpy())
        indices = [FORECAST_FEATURES.index(name) for name in ("current_temp", "current_o2")]
        np.testing.assert_allclose(predicted, observed[:, indices], rtol=1e-6, atol=1e-5)
        self.assertFalse(torch.allclose(prediction, y))
        path = self.root / "persistence.pth"
        torch.save(model.state_dict(), path)
        model.load_state_dict(torch.load(path, weights_only=True), strict=True)

    def test_forecast_smoke_all_models_without_test_loader(self):
        class WithoutTestLoader(ForecastDataPipeline):
            def build(self):
                super().build()
                self.test_loader = None
                return self

        with open(ROOT / "config" / "forecast_config.yaml") as handle:
            config = yaml.safe_load(handle)
        config["data"]["real_csv"] = str(self.path)
        config_path = self.root / "config.yaml"
        with open(config_path, "w") as handle:
            yaml.safe_dump(config, handle)
        output = self.root / "results"
        argv = [
            "benchmark", "--config", str(config_path), "--smoke-test",
            "--output_dir", str(output),
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(run_benchmark, "ForecastDataPipeline", WithoutTestLoader),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run_benchmark.main()
        for name in ("eblnn", "lnn", "lstm", "persistence"):
            seed_dir = output / "smoke" / name / "seed_42"
            result = json.loads((seed_dir / "results.json").read_text())
            self.assertEqual(result["protocol"], "forecast-real-v1")
            self.assertEqual(result["evaluation_split"], "validation")
            self.assertTrue(result["smoke_test"])
            self.assertEqual(result["protocol_metadata"]["horizon_recorded_steps"], 1)
            self.assertTrue(np.isfinite(list(result["test_metrics"].values())).all())
            self.assertIn("dropout_current_o2", result["disturbance"])
            self.assertIn("drift_bridgewall_temp_fast", result["extreme"])
            self.assertNotIn("dropout_air_fuel_ratio", result["disturbance"])
            self.assertTrue((seed_dir / "scalers.npz").is_file())
            torch.load(seed_dir / "models" / "best_model.pth", weights_only=True)
        runs = compare_results.load_model_runs(output / "smoke")
        report = compare_results.build_markdown(
            compare_results.build_summary(runs), protocol="forecast-real-v1", smoke_test=True,
        )
        self.assertIn("Real-Data Forecasting", report)
        self.assertIn("SMOKE TEST ONLY", report)
        self.assertNotIn("PINN-augmented", report)
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(run_benchmark, "create_trainer", side_effect=AssertionError("Unexpected training")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run_benchmark.main()
        first_seed = output / "smoke" / "eblnn" / "seed_42"
        for relative in ("models/best_model.pth", "history.npz", "scalers.npz"):
            artifact = first_seed / relative
            contents = artifact.read_bytes()
            artifact.write_bytes(b"")
            with (
                self.subTest(artifact=relative),
                mock.patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(ValueError, "missing or empty"),
            ):
                run_benchmark.main()
            artifact.write_bytes(contents)
        self.frame.loc[0, "OutletT"] += 1
        self.frame.to_csv(self.path, index=False)
        with (
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "source data may also have changed"),
        ):
            run_benchmark.main()


if __name__ == "__main__":
    unittest.main()
