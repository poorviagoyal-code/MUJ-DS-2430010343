"""Forecast next-year safely managed drinking-water access (%) per country.

Merges WHO water-access data with World Bank urban-population and
unemployment indicators, then compares Linear Regression, Random Forest and a
small Transformer on a time-based train/validation/test split.

Usage (from the repository root):
    python water_forecast.py
    python water_forecast.py --data-dir . --out-dir results --model-dir models --fig-dir figures
"""
import argparse
import os
import random

import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ["Water_Access", "Urban_Population", "Unemployment"]
WINDOW = 5


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_water(path):
    water = pd.read_csv(path)
    water = water[
        (water["Location type"] == "Country")
        & (water["Dim1"].astype(str).str.strip().str.lower() == "total")
    ].copy()
    water = water.rename(columns={
        "SpatialDimValueCode": "Country_Code",
        "Period": "Year",
        "FactValueNumeric": "Water_Access",
    })
    water["Year"] = pd.to_numeric(water["Year"], errors="coerce")
    water["Water_Access"] = pd.to_numeric(water["Water_Access"], errors="coerce")
    water = water.dropna(subset=["Country_Code", "Year", "Water_Access"])
    water["Year"] = water["Year"].astype(int)
    water["Country_Code"] = water["Country_Code"].astype(str).str.strip()
    water = water[water["Water_Access"].between(0, 100)]
    water = water[["Country_Code", "Year", "Water_Access"]]
    if water.duplicated(["Country_Code", "Year"]).any():
        raise ValueError("Duplicate water country-year rows found.")
    return water


def load_world_bank(path, column):
    """Reshape a World Bank wide CSV (4 metadata rows on top) to country-year."""
    df = pd.read_csv(path, skiprows=4)
    year_cols = [c for c in df.columns
                 if str(c).strip().isdigit() and len(str(c).strip()) == 4]
    if not year_cols:
        raise ValueError(f"No year columns found for {column}.")
    out = df.melt(id_vars=["Country Code"], value_vars=year_cols,
                  var_name="Year", value_name=column)
    out = out.rename(columns={"Country Code": "Country_Code"})
    out["Country_Code"] = out["Country_Code"].astype(str).str.strip()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["Country_Code", "Year"])
    out["Year"] = out["Year"].astype(int)
    out = out[out["Country_Code"].str.len() == 3]  # drop regional aggregates
    return out[["Country_Code", "Year", column]]


def build_dataset(data_dir):
    water = load_water(os.path.join(data_dir, "water.csv"))
    urban = load_world_bank(os.path.join(data_dir, "urban_population.csv"),
                            "Urban_Population")
    unemp = load_world_bank(os.path.join(data_dir, "unemployment.csv"),
                            "Unemployment")
    data = water.merge(urban, on=["Country_Code", "Year"], how="left",
                       validate="one_to_one")
    data = data.merge(unemp, on=["Country_Code", "Year"], how="left",
                      validate="one_to_one")
    return data.sort_values(["Country_Code", "Year"]).reset_index(drop=True)


def make_windows(data):
    data = data[data["Year"].between(2000, 2024)]
    X, y, years, countries = [], [], [], []
    for country, group in data.groupby("Country_Code"):
        group = group.sort_values("Year").dropna(subset=FEATURES).reset_index(drop=True)
        values = group[FEATURES].to_numpy(dtype=np.float32)
        yrs = group["Year"].to_numpy()
        for i in range(WINDOW, len(group)):
            if not np.all(np.diff(yrs[i - WINDOW:i + 1]) == 1):
                continue  # need consecutive years
            X.append(values[i - WINDOW:i])
            y.append(values[i, 0])
            years.append(yrs[i])
            countries.append(country)
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32),
            np.asarray(years, dtype=int), np.asarray(countries))


class WaterTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(len(FEATURES), 32)
        self.position = nn.Parameter(torch.randn(1, WINDOW, 32) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, dropout=0.1,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 1))

    def forward(self, x):
        x = self.embedding(x) + self.position
        x = self.encoder(x)
        return self.output(x[:, -1]).squeeze(-1)


def train_transformer(X_scaled, y_scaled, train, valid, device):
    model = WaterTransformer().to(device)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled[train]), torch.tensor(y_scaled[train])),
        batch_size=64, shuffle=True)
    X_val = torch.tensor(X_scaled[valid]).to(device)
    y_val = torch.tensor(y_scaled[valid]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_loss, best_weights, patience = float("inf"), None, 0
    for epoch in range(60):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: validation loss = {val_loss:.4f}")
        if patience >= 10:
            break
    model.load_state_dict(best_weights)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--fig-dir", default="figures")
    # parse_known_args ignores extra flags that Jupyter/Colab inject (e.g. -f kernel.json)
    args, _ = parser.parse_known_args()
    # Colab fallback: files uploaded to /content instead of the selected data directory
    if (not os.path.exists(os.path.join(args.data_dir, "water.csv"))
            and os.path.exists("/content/water.csv")):
        args.data_dir = "/content"
    for d in (args.out_dir, args.model_dir, args.fig_dir):
        os.makedirs(d, exist_ok=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data = build_dataset(args.data_dir)
    print("Merged dataset:", data.shape, "| countries:", data["Country_Code"].nunique())

    X, y, years, countries = make_windows(data)
    train = years <= 2018
    valid = (years >= 2019) & (years <= 2020)
    test = (years >= 2021) & (years <= 2024)
    if not (train.any() and valid.any() and test.any()):
        raise ValueError("A data split is empty. Check data coverage.")
    print(f"Train {train.sum()} | Validation {valid.sum()} | Test {test.sum()}")

    # Scalers are fit on training data only (no leakage)
    x_scaler = StandardScaler().fit(X[train].reshape(-1, len(FEATURES)))
    y_scaler = StandardScaler().fit(y[train].reshape(-1, 1))
    X_scaled = x_scaler.transform(X.reshape(-1, len(FEATURES))).reshape(X.shape).astype(np.float32)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    model = train_transformer(X_scaled, y_scaled, train, valid, device)

    X_flat = X_scaled.reshape(len(X_scaled), -1)
    linear = LinearRegression().fit(X_flat[train], y[train])
    forest = RandomForestRegressor(n_estimators=100, max_depth=12,
                                   random_state=42, n_jobs=-1).fit(X_flat[train], y[train])

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X_scaled[test]).to(device)).cpu().numpy()
    pred_transformer = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    predictions = {
        "Linear Regression": linear.predict(X_flat[test]),
        "Random Forest": forest.predict(X_flat[test]),
        "Transformer": pred_transformer,
        # Naive baseline: next year's value = last year's value (last window step)
        "Naive (repeat last year)": X[test][:, -1, 0],
    }
    results = pd.DataFrame([
        {"Model": name,
         "MAE": mean_absolute_error(y[test], p),
         "RMSE": np.sqrt(mean_squared_error(y[test], p)),
         "R2": r2_score(y[test], p)}
        for name, p in predictions.items()
    ])
    print(results.round(3).sort_values("RMSE").to_string(index=False))

    test_results = pd.DataFrame({
        "Country_Code": countries[test], "Year": years[test],
        "Actual": y[test], "Predicted": pred_transformer})

    plt.figure(figsize=(7, 6))
    plt.scatter(test_results["Actual"], test_results["Predicted"], alpha=0.6)
    lo = min(test_results["Actual"].min(), test_results["Predicted"].min())
    hi = max(test_results["Actual"].max(), test_results["Predicted"].max())
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel("Actual water access (%)")
    plt.ylabel("Predicted water access (%)")
    plt.title("Transformer: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(args.fig_dir, "03_transformer_actual_vs_predicted.png"), dpi=150)

    results.to_csv(os.path.join(args.out_dir, "model_comparison.csv"), index=False)
    test_results.to_csv(os.path.join(args.out_dir, "water_predictions.csv"), index=False)
    torch.save({
        "model_state_dict": model.state_dict(),
        "features": FEATURES, "window": WINDOW,
        "x_mean": x_scaler.mean_, "x_scale": x_scaler.scale_,
        "y_mean": y_scaler.mean_, "y_scale": y_scaler.scale_,
    }, os.path.join(args.model_dir, "water_transformer.pt"))
    print("Saved results, plot and model.")


if __name__ == "__main__":
    main()
