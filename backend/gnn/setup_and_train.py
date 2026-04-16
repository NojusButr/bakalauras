#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


def check_deps():
    """Check and install PyTorch Geometric."""
    try:
        import torch_geometric
        print(f" PyTorch Geometric {torch_geometric.__version__} already installed")
        return True
    except ImportError:
        print("Installing PyTorch Geometric...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch-geometric",
                "--quiet"
            ])
            print(" PyTorch Geometric installed")
            return True
        except subprocess.CalledProcessError:
            print(" Failed to install PyTorch Geometric")
            print("  Try manually: pip install torch-geometric")
            return False


def check_torch():
    try:
        import torch
        cuda = "CUDA available" if torch.cuda.is_available() else "CPU only"
        print(f" PyTorch {torch.__version__} ({cuda})")
        return True
    except ImportError:
        print("PyTorch not found. Install it first:")
        print("  pip install torch")
        return False


def main():
    project_root = Path(__file__).parent.parent
    gnn_dir = Path(__file__).parent

    print("=" * 60)
    print("GNN Pipeline Setup")
    print("=" * 60)

    if not check_torch():
        return
    if not check_deps():
        return

    city_dir = project_root / "cities" / "vilnius"
    graph_path = city_dir / "graph.pkl"

    if not graph_path.exists():
        print(f"\n✗ Graph not found at {graph_path}")
        print("  Start the FastAPI server first to generate city data:")
        print("  uvicorn app.main:app --reload")
        print("  Then visit http://localhost:8000/cities/vilnius")
        return

    print(f"✓ Graph found at {graph_path}")

    # Check snapshots
    snapshots_dir = city_dir / "snapshots"
    snapshot_count = len(list(snapshots_dir.glob("*.json"))) if snapshots_dir.exists() else 0
    print(f"  Found {snapshot_count} traffic snapshot(s)")

    if snapshot_count == 0:
        print("\n No snapshots found. The model will train on road structure only.")
        print("  For better results, collect some snapshots first:")
        print("  curl -X POST http://localhost:8000/traffic/snapshot/vilnius")
        print("  (Repeat at different times of day for more training data)")

    # Train
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

    output_dir = project_root / "models"
    output_dir.mkdir(exist_ok=True)

    sys.path.insert(0, str(gnn_dir))
    from train import main as train_main

    sys.argv = [
        "train.py",
        "--city-dir", str(city_dir),
        "--output-dir", str(output_dir),
        "--epochs", "200",
        "--hidden-dim", "64",
        "--num-layers", "3",
    ]

    train_main()

    model_path = output_dir / "best_model.pt"
    if model_path.exists():
        print(f"\n{'='*60}")
        print(f"✓ Model saved to {model_path}")
        print(f"  The /route/gnn and /route/compare endpoints are now available!")
        print(f"{'='*60}")
    else:
        print("\n✗ Training failed — no model produced")


if __name__ == "__main__":
    main()
