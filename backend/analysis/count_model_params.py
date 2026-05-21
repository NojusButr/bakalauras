"""
Count and print the number of parameters for both models.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from gnn.gnn_model import EdgeTravelTimeGNN
from gnn.gnn_route_classifier import RouteClassifierGNN


def count_parameters(model):
    """Count total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=" * 70)
    print("MODEL PARAMETER COUNTING")
    print("=" * 70)
    print("Using hyperparameters from training files:")
    print("  EdgeTravelTimeGNN: hidden_dim=64, num_layers=3 (from setup_and_train.py)")
    print("  RouteClassifierGNN: hidden_dim=64, num_steps=8 (from train_route_classifier.py)")
    print()
    
    # Initialize both models with hyperparameters
    model1 = EdgeTravelTimeGNN(
        node_features=5,
        edge_features=12,
        hidden_dim=64,
        num_layers=3,
        dropout=0.1
    )
    
    model2 = RouteClassifierGNN(
        node_features=9,
        edge_features=12,
        hidden_dim=64,
        num_steps=8,
        dropout=0.1
    )
    
    # Count parameters
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)
    total_params = params1 + params2
    
    # Print results
    print(f"\nModel 1: EdgeTravelTimeGNN")
    print(f"  Parameters: {params1:,}")
    
    print(f"\nModel 2: RouteClassifierGNN")
    print(f"  Parameters: {params2:,}")
    
    print(f"\nTotal Parameters (both models): {total_params:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
