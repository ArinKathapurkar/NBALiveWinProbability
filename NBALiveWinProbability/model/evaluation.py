"""
evaluate.py — Model evaluation for LiveProbability win probability model.

Run after training:
    python model/evaluate.py

Produces:
    - Accuracy vs home-team baseline
    - Log loss vs reference benchmarks
    - Calibration plot (saved to model/calibration.png)
    - Per-period accuracy breakdown
    - Score differential accuracy breakdown
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# ── paths (mirrors model.py) ────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "S23": ROOT / "data" / "historical" / "pbp_2023_24.csv",
    "S24": ROOT / "data" / "historical" / "pbp_2024_25.csv",
    "S25": ROOT / "data" / "historical" / "pbp_2025_26.csv",
}

TRAINING = {
    "scaler":     ROOT / "model" / "scaler.pkl",
    "checkpoint": ROOT / "model" / "checkpoints" / "win_prob.pt",
    "cal_plot":   ROOT / "model" / "calibration.png",
    "temperature": ROOT / "model" / "checkpoints" / "temperature.pt"
}

FEATURE_COLS = [
    'period',
    'seconds_remaining_in_period',
    'seconds_remaining_in_game',
    'score_diff',
    'home_fouls',
    'away_fouls',
    'foul_diff',
    'possession',
    'lead_time_interaction',
    'home_in_bonus', 
    'away_in_bonus' 
]

# ── model definition (must match model.py exactly) ──────────────────────────

class LiveProbability(nn.Module):
    """
    Predicts probability that the HOME team wins.
    Input:  8 features (see FEATURE_COLS)
    Output: float in (0, 1)
    """
    def __init__(self):
        super().__init__()
        self.layer1  = nn.Linear(11, 32)
        self.relu    = nn.ReLU()
        self.layer2  = nn.Linear(32, 16)
        self.layer3  = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x


class TemperatureScaler(nn.Module):
    """
    Wraps a trained model and learns a single temperature
    parameter T that scales logits before sigmoid.
    T > 1 makes the model less confident.
    T < 1 makes it more confident.
    Train this on val set AFTER main training is done.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # get raw logits before sigmoid
        with torch.no_grad():
            logits = self.model.layer3(
                self.model.relu(
                    self.model.layer2(
                        self.model.relu(
                            self.model.layer1(x)
                        )
                    )
                )
            )
        return torch.sigmoid(logits / self.temperature)

# ── helpers ──────────────────────────────────────────────────────────────────

def load_model_and_scaler():
    """Load trained model and fitted scaler from disk."""
    for name, path in TRAINING.items():
        if name in ("scaler", "checkpoint") and not path.exists():
            raise FileNotFoundError(
                f"Missing {name} at {path}. "
                f"Have you run model.py to train first?"
            )

    model = LiveProbability()
    model.load_state_dict(torch.load(TRAINING["checkpoint"], map_location="cpu"))
    model.eval()

    scaler = joblib.load(TRAINING["scaler"])
    return model, scaler


def load_val_data(scaler):
    """
    Reproduce the exact same train/val split as model.py.
    Returns (val_df, X_val_t, y_val_t).
    """
    frames = [pd.read_csv(p) for p in PATHS.values() if p.exists()]
    if not frames:
        raise FileNotFoundError("No CSV data found — run fetch_historical.py first")

    data = pd.concat(frames, ignore_index=True)
    
    data['lead_time_interaction'] = data['score_diff'] * data['seconds_remaining_in_game']
    data['home_in_bonus'] = (data['home_fouls'] >= 5).astype(int)
    data['away_in_bonus'] = (data['away_fouls'] >= 5).astype(int)

    # reproduce the same split — seed must match model.py
    all_games = data['game_id'].unique()
    np.random.seed(42)
    np.random.shuffle(all_games)

    split    = int(0.8 * len(all_games))
    val_ids  = all_games[split:]
    val_df   = data[data['game_id'].isin(val_ids)].copy()

    X_val = scaler.transform(val_df[FEATURE_COLS].to_numpy())
    y_val = val_df['home_win'].to_numpy()

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    return val_df, X_val_t, y_val_t


def get_probs_and_actuals(model, X_val_t, y_val_t):
    """Run a single forward pass over the full val set."""
    with torch.no_grad():
        probs = model(X_val_t).numpy().flatten()
    actuals = y_val_t.numpy().flatten()
    return probs, actuals


# ── evaluation functions ─────────────────────────────────────────────────────

def eval_accuracy(probs, actuals):
    """
    Binary accuracy at 0.5 threshold, compared to the naive
    home-team-always-wins baseline (~57% in the NBA).
    """
    print("\n── Accuracy ───────────────────────────────────────────")
    predicted = (probs >= 0.5).astype(float)
    accuracy  = (predicted == actuals).mean()

    home_baseline = actuals.mean()  # fraction of games home team won

    print(f"  Model accuracy:          {accuracy:.2%}")
    print(f"  Home-team baseline:      {home_baseline:.2%}")
    print(f"  Improvement over base:   {accuracy - home_baseline:+.2%}")

    if accuracy < home_baseline:
        print("  ⚠️  Model is WORSE than always predicting home team wins.")
    elif accuracy < home_baseline + 0.03:
        print("  Model is marginally better than baseline — consider more data or features.")
    else:
        print("  Model meaningfully outperforms baseline.")


def eval_log_loss(probs, actuals):
    """
    Log loss vs reference benchmarks.
    Lower is better. Random = 0.693, good model ≈ 0.500.
    """
    print("\n── Log Loss ───────────────────────────────────────────")
    ll = log_loss(actuals, probs)

    benchmarks = {
        "Random (always 0.5)":      0.693,
        "Always predicts home win": log_loss(actuals, np.full_like(actuals, actuals.mean())),
        "Decent model":             0.550,
        "Good model":               0.500,
        "Vegas / ESPN level":       0.450,
    }

    print(f"  Your model log loss: {ll:.4f}\n")
    print(f"  {'Benchmark':<30} {'Log Loss':>10}  {'vs Yours':>10}")
    print(f"  {'-'*52}")
    for name, val in benchmarks.items():
        diff = ll - val
        flag = "✅ better" if diff < 0 else "❌ worse"
        print(f"  {name:<30} {val:>10.4f}  {flag}")


def eval_calibration(probs, actuals, n_buckets=10):
    """
    Calibration plot: predicted probability vs actual win rate per bucket.
    A perfectly calibrated model follows the diagonal exactly.
    Saved to model/calibration.png.
    """
    print("\n── Calibration ────────────────────────────────────────")

    buckets        = np.linspace(0, 1, n_buckets + 1)
    centers, rates, counts = [], [], []

    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        mask   = (probs >= lo) & (probs < hi)
        n      = mask.sum()
        if n > 0:
            centers.append((lo + hi) / 2)
            rates.append(actuals[mask].mean())
            counts.append(n)

    # print table
    print(f"\n  {'Pred bucket':<15} {'Actual win rate':>16} {'N plays':>10}")
    print(f"  {'-'*43}")
    for c, r, n in zip(centers, rates, counts):
        gap  = abs(r - c)
        flag = " ⚠️" if gap > 0.05 else ""
        print(f"  {c:.0%} — {c+0.1:.0%}       {r:>14.1%}   {n:>9,}{flag}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    ax.plot(centers, rates, 'o-', color='#5DCAA5', linewidth=2,
            markersize=7, label='Your model')
    ax.fill_between(centers,
                    [max(0, r - 0.05) for r in rates],
                    [min(1, r + 0.05) for r in rates],
                    alpha=0.15, color='#5DCAA5')

    ax.set_xlabel('Predicted probability (home win)', fontsize=12)
    ax.set_ylabel('Actual home win rate', fontsize=12)
    ax.set_title('Calibration Plot — LiveProbability Model', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    out = TRAINING["cal_plot"]
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Calibration plot saved → {out}")


def eval_by_period(model, val_df, scaler):
    """
    Accuracy broken down by period.
    Q4 accuracy should be significantly higher than Q1.
    If not, the model hasn't learned that time remaining matters.
    """
    print("\n── Accuracy by Period ─────────────────────────────────")
    print(f"  {'Period':<10} {'Accuracy':>10} {'N plays':>10}")
    print(f"  {'-'*32}")

    for period in sorted(val_df['period'].unique()):
        mask    = val_df['period'] == period
        subset  = val_df[mask]
        X       = torch.tensor(
                      scaler.transform(subset[FEATURE_COLS].to_numpy()),
                      dtype=torch.float32
                  )
        y       = subset['home_win'].to_numpy()

        with torch.no_grad():
            preds = (model(X).numpy().flatten() >= 0.5).astype(float)

        acc   = (preds == y).mean()
        label = f"Q{period}" if period <= 4 else f"OT{period - 4}"
        print(f"  {label:<10} {acc:>10.2%} {len(y):>10,}")


def eval_by_score_diff(model, val_df, scaler):
    """
    Accuracy broken down by score differential bucket.
    Large leads should have very high accuracy.
    Close games (within 5) should be near 50/50 — that's correct behavior,
    not a model failure.
    """
    print("\n── Accuracy by Score Differential ────────────────────")
    print(f"  {'Score diff':<18} {'Accuracy':>10} {'N plays':>10}")
    print(f"  {'-'*40}")

    buckets = [
        ("Blowout (>15)",    lambda d: d >  15),
        ("Large lead (10-15)", lambda d: (d >= 10) & (d <= 15)),
        ("Moderate (5-9)",   lambda d: (d >= 5)  & (d <= 9)),
        ("Close (0-4)",      lambda d: (d >= -4) & (d <= 4)),
        ("Trailing (5-9)",   lambda d: (d >= -9) & (d <= -5)),
        ("Down big (10-15)", lambda d: (d >= -15) & (d <= -10)),
        ("Way down (<-15)",  lambda d: d < -15),
    ]

    diffs = val_df['score_diff']

    for label, condition in buckets:
        mask   = condition(diffs)
        subset = val_df[mask]
        if len(subset) == 0:
            continue

        X = torch.tensor(
                scaler.transform(subset[FEATURE_COLS].to_numpy()),
                dtype=torch.float32
            )
        y = subset['home_win'].to_numpy()

        with torch.no_grad():
            preds = (model(X).numpy().flatten() >= 0.5).astype(float)

        acc = (preds == y).mean()
        print(f"  {label:<18} {acc:>10.2%} {len(y):>10,}")
        
        
        

def fit_temperature_scaler(model, X_val_t, y_val_t):
    """
    Fits a temperature scaler on the validation set and saves it to disk.
 
    Steps:
        1. Wraps the trained model in TemperatureScaler (T starts at 1.0)
        2. Optimizes T using BCE loss on the val set
        3. Prints before/after calibration comparison
        4. Saves the fitted T to TRAINING["temperature"]
        5. Returns the fitted TemperatureScaler
 
    Call this after eval_calibration() so you can see the before state first.
    Then call eval_calibration() again with the scaled probs to verify it worked.
    """
    print("\n── Temperature Scaling ────────────────────────────────")
 
    scaler_model = TemperatureScaler(model)
    optimizer    = torch.optim.LBFGS(
        [scaler_model.temperature],
        lr=0.01,
        max_iter=500
    )
    loss_fn = nn.BCELoss()
 
    # LBFGS requires a closure — it evaluates the loss multiple times per step
    def closure():
        optimizer.zero_grad()
        preds = scaler_model(X_val_t)
        loss  = loss_fn(preds, y_val_t)
        loss.backward()
        return loss
 
    print("  Fitting temperature parameter...")
    optimizer.step(closure)
 
    T = scaler_model.temperature.item()
    print(f"  Fitted temperature T = {T:.4f}")
 
    if T > 1.05:
        print(f"  Model was overconfident — T={T:.2f} pulls probabilities toward 0.5")
    elif T < 0.95:
        print(f"  Model was underconfident — T={T:.2f} pushes probabilities toward 0 or 1")
    else:
        print(f"  Model was well calibrated — T≈1.0, minimal adjustment needed")
 
    # before vs after comparison
    with torch.no_grad():
        probs_before = model(X_val_t).numpy().flatten()
        probs_after  = scaler_model(X_val_t).numpy().flatten()
 
    actuals = y_val_t.numpy().flatten()
 
    ll_before = log_loss(actuals, probs_before)
    ll_after  = log_loss(actuals, probs_after)
 
    print(f"\n  {'':30} {'Log Loss':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Before temperature scaling':<30} {ll_before:>10.4f}")
    print(f"  {'After temperature scaling':<30} {ll_after:>10.4f}")
    print(f"  {'Improvement':<30} {ll_before - ll_after:>+10.4f}")
 
    # save the temperature value
    TRAINING["temperature"].parent.mkdir(parents=True, exist_ok=True)
    torch.save({"temperature": T}, TRAINING["temperature"])
    print(f"\n  Temperature saved → {TRAINING['temperature']}")
 
    return scaler_model, probs_after





# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading model and scaler...")
    model, scaler = load_model_and_scaler()

    print("Loading validation data...")
    val_df, X_val_t, y_val_t = load_val_data(scaler)
    print(f"Validation set: {len(val_df):,} rows from {val_df['game_id'].nunique():,} games")

    probs, actuals = get_probs_and_actuals(model, X_val_t, y_val_t)

    eval_accuracy(probs, actuals)
    eval_log_loss(probs, actuals)

    print("\nCalibration BEFORE temperature scaling:")
    eval_calibration(probs, actuals)

    scaled_model, probs_scaled = fit_temperature_scaler(model, X_val_t, y_val_t)

    print("\nCalibration AFTER temperature scaling:")
    eval_calibration(probs_scaled, actuals)

    eval_by_period(scaled_model, val_df, scaler)
    eval_by_score_diff(scaled_model, val_df, scaler)

    print("\n── Done ───────────────────────────────────────────────\n")