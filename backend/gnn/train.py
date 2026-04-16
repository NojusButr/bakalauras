"""
Training approach:
  - Dataset split:
    - Training: snapshots from earlier dates + their augmented variants
    - Validation: snapshots from later dates (never seen during training)
  - Loss = MSE on log(travel_time) to handle the wide range
  - Adam optimizer with cosine LR schedule
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from data_pipeline import build_dataset, load_all_snapshots
from gnn_model import EdgeTravelTimeGNN


def log_mse_loss(pred, target):
    """MSE on log-transformed values — handles wide travel time range."""
    return F.mse_loss(torch.log1p(pred), torch.log1p(target))


def mape_metric(pred, target, eps=1.0):
    mask = target > eps
    if mask.sum() == 0:
        return 0.0
    return (torch.abs(pred[mask] - target[mask]) / target[mask]).mean().item() * 100


def mae_metric(pred, target):
    return torch.abs(pred - target).mean().item()


def train_epoch(model, data_list, optimizer, device):
    """Train on ALL edges of training snapshots."""
    model.train()
    total_loss = 0.0
    total_edges = 0

    for data in data_list:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.edge_attr)
        loss = log_mse_loss(pred, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * data.edge_index.shape[1]
        total_edges += data.edge_index.shape[1]

    return total_loss / max(total_edges, 1)


@torch.no_grad()
def eval_epoch(model, data_list, device):
    """Evaluate on ALL edges of validation snapshots."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mape = 0.0
    total_edges = 0
    n_samples = 0

    for data in data_list:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.edge_attr)
        ne = data.edge_index.shape[1]

        total_loss += log_mse_loss(pred, data.y).item() * ne
        total_mae += mae_metric(pred, data.y) * ne
        total_mape += mape_metric(pred, data.y)
        total_edges += ne
        n_samples += 1

    n = max(total_edges, 1)
    return {
        "loss": total_loss / n,
        "mae": total_mae / n,
        "mape": total_mape / max(n_samples, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train GNN travel time predictor")
    parser.add_argument("--city-dir", type=str, default="cities/vilnius")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of snapshots used for validation (by date)")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild dataset cache")
    args = parser.parse_args()

    city_dir = Path(args.city_dir)
    graph_path = city_dir / "graph.pkl"
    output_dir = city_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not graph_path.exists():
        print(f"ERROR: Graph not found at {graph_path}")
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"\n{'='*60}")
    print(f"Loading data from {city_dir}")
    print(f"{'='*60}")
    dataset = build_dataset(graph_path, city_dir, force_rebuild=args.rebuild)

    if not dataset:
        print("No data to train on!")
        return

    snapshot_files = load_all_snapshots(city_dir)
    n_snapshots = len(snapshot_files)
    variants_per_snap = len(dataset) // max(n_snapshots, 1)
    
    if variants_per_snap < 1:
        variants_per_snap = 1

    n_val_snaps = max(1, int(n_snapshots * args.val_ratio))
    n_train_snaps = n_snapshots - n_val_snaps

    train_data = dataset[:n_train_snaps * variants_per_snap]
    val_data = dataset[n_train_snaps * variants_per_snap:]

    if not val_data:
        val_data = train_data[-2:]
        train_data = train_data[:-2]

    print(f"\nSnapshot-level split:")
    print(f"  Total snapshots: {n_snapshots} ({variants_per_snap} variants each = {len(dataset)} samples)")
    print(f"  Train: {n_train_snaps} snapshots ({len(train_data)} samples)")
    print(f"  Val:   {n_val_snaps} snapshots ({len(val_data)} samples) — UNSEEN during training")

    d = dataset[0]
    num_edges = d.edge_index.shape[1]
    print(f"\nGraph: {d.x.shape[0]} nodes, {num_edges} edges")
    print(f"Node features: {d.x.shape[1]}, Edge features: {d.edge_attr.shape[1]}")
    print(f"Travel time range: {d.y.min():.1f}s — {d.y.max():.1f}s (mean {d.y.mean():.1f}s)")

    model = EdgeTravelTimeGNN(
        node_features=d.x.shape[1],
        edge_features=d.edge_attr.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs")
    print(f"{'='*60}")

    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_data, optimizer, device)
        val_metrics = eval_epoch(model, val_data, device)
        scheduler.step()

        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_mape": val_metrics["mape"],
            "lr": optimizer.param_groups[0]["lr"],
            "time": elapsed,
        }
        history.append(record)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "node_features": d.x.shape[1],
                    "edge_features": d.edge_attr.shape[1],
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                },
                "epoch": epoch,
                "val_loss": best_val_loss,
                "val_mae": val_metrics["mae"],
                "val_mape": val_metrics["mape"],
                "n_train_snapshots": n_train_snaps,
                "n_val_snapshots": n_val_snaps,
            }, output_dir / "best_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{args.epochs} | "
                f"train={train_loss:.4f} | "
                f"val={val_metrics['loss']:.4f} | "
                f"MAE={val_metrics['mae']:.1f}s | "
                f"MAPE={val_metrics['mape']:.1f}% | "
                f"lr={optimizer.param_groups[0]['lr']:.6f} | "
                f"{elapsed:.1f}s"
            )

    print(f"\n{'='*60}")
    print(f"Training complete! Best val_loss={best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Model saved to {output_dir / 'best_model.pt'}")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"Final evaluation on validation set (unseen snapshots)")
    print(f"{'='*60}")

    ckpt = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    final_metrics = eval_epoch(model, val_data, device)
    print(f"  Val Loss: {final_metrics['loss']:.4f}")
    print(f"  MAE: {final_metrics['mae']:.1f} seconds")
    print(f"  MAPE: {final_metrics['mape']:.1f}%")

    # Sample predictions from a val snapshot
    print(f"\nSample predictions (first 10 edges from val snapshot):")
    model.eval()
    with torch.no_grad():
        data = val_data[0].to(device)
        pred = model(data.x, data.edge_index, data.edge_attr)
        indices = torch.randperm(data.edge_index.shape[1])[:10]
        for idx in indices:
            p = pred[idx].item()
            t = data.y[idx].item()
            print(f"  Edge {idx.item()}: predicted={p:.1f}s, actual={t:.1f}s, error={abs(p-t):.1f}s")


if __name__ == "__main__":
    main()