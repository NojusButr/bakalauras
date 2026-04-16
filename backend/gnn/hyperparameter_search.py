"""
Uses only clean samples (1 per snapshot) and 15 epochs per config
to quickly find the best combo. 
"""

import json
import time
import itertools
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline import graph_to_pyg_data, load_all_snapshots
from gnn_model import EdgeTravelTimeGNN


def log_mse_loss(pred, target):
    return F.mse_loss(torch.log1p(pred), torch.log1p(target))


def quick_train(train_data, val_data, config, device, epochs=15):
    """Train quickly, return best val loss."""
    d = train_data[0]
    model = EdgeTravelTimeGNN(
        node_features=d.x.shape[1],
        edge_features=d.edge_attr.shape[1],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        for data in train_data:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index, data.edge_attr)
            loss = log_mse_loss(pred, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # Val every 5 epochs
        if epoch % 5 == 4 or epoch == epochs - 1:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_data:
                    data = data.to(device)
                    pred = model(data.x, data.edge_index, data.edge_attr)
                    val_loss += log_mse_loss(pred, data.y).item()
            val_loss /= max(len(val_data), 1)
            best_val = min(best_val, val_loss)

    return best_val


def main():
    project_root = Path(__file__).parent.parent
    city_dir = project_root / "cities" / "vilnius"
    graph_path = city_dir / "graph.pkl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading clean snapshots only (no augmented variants)...")
    snapshot_files = load_all_snapshots(city_dir)
    print(f"Found {len(snapshot_files)} snapshots")

    clean_data = []
    for sf in snapshot_files:
        clean_data.append(graph_to_pyg_data(graph_path, snapshot_path=sf))
    print(f"Loaded {len(clean_data)} clean samples")


    n_val = max(1, int(len(clean_data) * 0.2))
    train_data = clean_data[:-n_val]
    val_data = clean_data[-n_val:]
    print(f"Train: {len(train_data)} snapshots, Val: {len(val_data)} snapshots")


    param_grid = {
        "hidden_dim": [256],
        "num_layers": [4, 5, 6],
        "lr": [0.05],
        "dropout": [0.2, 0.25, 0.3, 0.4],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    EPOCHS = 15

    print(f"\nTesting {len(combos)} configs × {EPOCHS} epochs each")
    print("=" * 70)

    results = []
    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        t0 = time.time()

        try:
            val_loss = quick_train(train_data, val_data, config, device, epochs=EPOCHS)
            elapsed = time.time() - t0
            results.append({**config, "val_loss": val_loss, "time": round(elapsed, 1)})
            print(f"  [{i+1:2d}/{len(combos)}] h={config['hidden_dim']:3d} "
                  f"L={config['num_layers']} lr={config['lr']:.4f} "
                  f"do={config['dropout']:.2f} → val={val_loss:.4f} ({elapsed:.0f}s)")
        except Exception as e:
            print(f"  [{i+1:2d}/{len(combos)}] FAILED: {e}")
            results.append({**config, "val_loss": float("inf"), "time": 0})

    results.sort(key=lambda x: x["val_loss"])

    print("\n" + "=" * 70)
    print("TOP 10:")
    print("=" * 70)
    for i, r in enumerate(results[:10]):
        print(f"  #{i+1}: hidden={r['hidden_dim']}, layers={r['num_layers']}, "
              f"lr={r['lr']}, dropout={r['dropout']:.2f} → val={r['val_loss']:.4f}")

    output_path = city_dir / "models" / "hyperparam_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    best = results[0]
    print(f"\n{'=' * 70}")
    print(f"BEST: hidden={best['hidden_dim']}, layers={best['num_layers']}, "
          f"lr={best['lr']}, dropout={best['dropout']}")
    print(f"\nTrain with full augmented dataset:")
    print(f"  python gnn/train.py --city-dir cities/vilnius "
          f"--hidden-dim {best['hidden_dim']} --num-layers {best['num_layers']} "
          f"--lr {best['lr']} --dropout {best['dropout']} --epochs 200")


if __name__ == "__main__":
    main()