"""Quick diagnostic: why does PINN inference give OutletT ≈ 147 instead of ≈ 324?"""
import torch, pickle, numpy as np, pandas as pd
from train_pinn_v2 import FurnacePINN_v2, CONSTANTS

# 1. Load checkpoint
ckpt = torch.load("checkpoints_v2/best_pinn_v2.pth", map_location="cpu", weights_only=False)
print("=== CHECKPOINT ===")
print("Keys:", list(ckpt.keys()))
print("Best epoch:", ckpt["epoch"])
print("Val loss:", ckpt["val_loss"])
print("Saved theta_eff (raw param value):", ckpt["theta_eff"])
print("Saved afr_stoich (raw param value):", ckpt["afr_stoich"])
print("var_T:", ckpt["var_T"])
print("var_O2:", ckpt["var_O2"])
print("feature_cols:", ckpt["feature_cols"])

# 2. Reconstruct model
model = FurnacePINN_v2(input_dim=len(ckpt["feature_cols"]))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("\n=== MODEL ===")
print("theta_eff raw param:", model.theta_eff.item())
print("afr_stoich raw param:", model.afr_stoich.item())

# Last linear layer
last = list(model.net.children())[-1]
print("Last layer bias:", last.bias.data.numpy())

# 3. Load scaler
with open("checkpoints_v2/scaler_v2.pkl", "rb") as f:
    scaler = pickle.load(f)
print("\n=== SCALER ===")
print("mean_:", scaler.mean_)
print("scale_:", scaler.scale_)
print("n_features:", scaler.n_features_in_)

# 4. Test on real data
df = pd.read_csv("cleaned_furnace_data.csv")
fcols = ckpt["feature_cols"]
X_real = df[fcols].values[:10]
X_scaled = scaler.transform(X_real)

print("\n=== REAL INPUT (first 3 rows) ===")
for i in range(3):
    print(f"  raw:    {X_real[i]}")
    print(f"  scaled: {X_scaled[i]}")

with torch.no_grad():
    preds = model(torch.tensor(X_scaled, dtype=torch.float32))
    print("\n=== MODEL PREDICTIONS (first 10) ===")
    print("  [OutletT, ExcessO2]")
    for i in range(10):
        print(f"  pred: {preds[i].numpy()}   real: [{df['OutletT'].iloc[i]:.2f}, {df['ExcessO2'].iloc[i]:.2f}]")

# 5. Also test with training-split data (first 70%)
n = len(df)
train_end = int(n * 0.70)
X_train_raw = df[fcols].values[:train_end]
y_train = df[["OutletT", "ExcessO2"]].values[:train_end]
X_train_s = scaler.transform(X_train_raw)

# Predictions on a random train subset
rng = np.random.default_rng(0)
idx = rng.choice(train_end, 5, replace=False)
with torch.no_grad():
    sub = torch.tensor(X_train_s[idx], dtype=torch.float32)
    p = model(sub).numpy()
    print("\n=== RANDOM TRAIN SAMPLES ===")
    for i, ix in enumerate(idx):
        print(f"  row {ix}: pred=[{p[i,0]:.2f}, {p[i,1]:.2f}]  real=[{y_train[ix,0]:.2f}, {y_train[ix,1]:.2f}]")

# 6. Mean prediction over 1000 random train rows
idx1k = rng.choice(train_end, 1000, replace=False)
with torch.no_grad():
    p1k = model(torch.tensor(X_train_s[idx1k], dtype=torch.float32)).numpy()
    print(f"\nMean pred over 1000 train rows: OutletT={p1k[:,0].mean():.2f}  ExcessO2={p1k[:,1].mean():.2f}")
    print(f"Std  pred over 1000 train rows: OutletT={p1k[:,0].std():.2f}  ExcessO2={p1k[:,1].std():.2f}")
    print(f"Real mean over same rows:        OutletT={y_train[idx1k,0].mean():.2f}  ExcessO2={y_train[idx1k,1].mean():.2f}")
