"""Quick 2-epoch smoke test for the ablation pipeline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_real import RealDataPipeline
from src.model import create_model
from src.sampler import build_sampler
from src.train import GenerativeTrainer

data_dir = ROOT.parent / "dataset"

print("=" * 60)
print("  SMOKE TEST — 2-epoch ablation pipeline")
print("=" * 60)

# 1. Data pipeline
pipe = RealDataPipeline(
    real_csv=str(data_dir / "real_furnace_eblnn.csv"),
    edge_csv=str(data_dir / "edge_cases_v2_eblnn.csv"),
    seq_len=30,
    batch_size=64,
    seed=42,
).build()

# 2. Model
model = create_model(
    input_size=5, hidden_size=128,
    ebm_hidden_dims=[128, 64], device="cpu",
)

# 3. Sampler (fewer steps for speed)
sampler = build_sampler(model, n_steps=5, step_size=0.01, noise_scale=0.005)

# 4. Trainer
trainer = GenerativeTrainer(
    model=model,
    sampler=sampler,
    config={
        "epochs": 2,
        "learning_rate": 0.001,
        "alpha": 1.0,
        "l2_reg": 0.1,
        "energy_clamp": 20.0,
        "patience": 100,
        "min_delta": 0.0001,
        "early_stopping": False,
        "buffer_capacity": 1000,
        "buffer_prob": 0.95,
        "seq_len": 30,
        "input_size": 5,
    },
    device="cpu",
    use_wandb=False,
)

# 5. Train (2 epochs)
trainer.train(
    pipe.train_loader, pipe.val_loader,
    save_path="/tmp/ablation_smoke",
)

# 6. Evaluate
metrics = trainer.evaluate(pipe.test_loader, target_scaler=pipe.target_scaler)

print("\n" + "=" * 60)
print(f"  SMOKE TEST PASSED")
print(f"  Temp RMSE : {metrics['test_rmse_temp']:.4f} °C")
print(f"  O2   RMSE : {metrics['test_rmse_o2']:.4f} %")
print("=" * 60)
