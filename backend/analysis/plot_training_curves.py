
"""
Create training curves for ML models
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_curves():
    """
    Load and visualize training curves for Model 1 (GraphSAGE - Travel Time Prediction).
    """
    
    script_dir = Path(__file__).parent.parent
    # Load Model 1 training history
    history_file = script_dir / 'models' / 'training_history.json'
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    
    # Find best epoch (minimum val loss)
    best_epoch = min(range(len(val_loss)), key=lambda i: val_loss[i])
    best_val_loss = val_loss[best_epoch]
    best_epoch_num = epochs[best_epoch]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ===== MODEL 1: GraphSAGE Travel Time Prediction =====
    ax = axes[0]
    
    ax.plot(epochs, train_loss, 'o-', linewidth=2, markersize=4, label='Train Loss', color='#2E86AB')
    ax.plot(epochs, val_loss, 's-', linewidth=2, markersize=4, label='Validation Loss', color='#A23B72')
    
    # Mark best epoch
    ax.scatter([best_epoch_num], [best_val_loss], s=200, color='gold', marker='*', 
              zorder=5, label=f'Best (Epoch {best_epoch_num}, Loss={best_val_loss:.5f})', edgecolor='black', linewidth=1.5)
    ax.axvline(best_epoch_num, color='gold', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (Log-MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Model 1: GraphSAGE Edge Travel Time Prediction\n(200 Epochs, MSE Loss on log-transformed targets)', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add statistics box
    train_final = train_loss[-1]
    val_final = val_loss[-1]
    train_improved = (train_loss[0] - train_final) / train_loss[0] * 100
    val_improved = (val_loss[0] - val_final) / val_loss[0] * 100
    
    stats_text = (
        f'Final Train Loss: {train_final:.6f}\n'
        f'Final Val Loss: {val_final:.6f}\n'
        f'Train Improvement: {train_improved:.1f}%\n'
        f'Val Improvement: {val_improved:.1f}%\n\n'
        f'Overfitting Analysis:\n'
        f'Gap at Best: {best_val_loss - train_loss[best_epoch]:.6f}\n'
        f'Gap at End: {val_final - train_final:.6f}'
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=1.5))
    
    # ===== MODEL 2: LSTM-GN Route Classifier (ACTUAL DATA) =====
    ax = axes[1]
    
    # Load Model 2 actual training history
    classifier_history_file = script_dir / 'cities' / 'vilnius' / 'models' / 'classifier_history.json'
    with open(classifier_history_file, 'r') as f:
        classifier_history = json.load(f)
    
    epochs_2 = [h['epoch'] + 1 for h in classifier_history]  # Convert 0-indexed to 1-indexed
    train_loss_2 = [h['train_loss'] for h in classifier_history]
    val_loss_2 = [h['val_loss'] for h in classifier_history]
    train_f1 = [h['f1'] for h in classifier_history]
    
    # For classifier, we'll plot using val_loss on secondary metric
    # But compute a "validation F1" proxy based on the pattern
    # Actually, let's plot both train and val loss for consistency
    best_epoch_2 = min(range(len(val_loss_2)), key=lambda i: val_loss_2[i])
    best_val_loss_2 = val_loss_2[best_epoch_2]
    best_epoch_num_2 = epochs_2[best_epoch_2]
    
    ax.plot(epochs_2, train_loss_2, 'o-', linewidth=2, markersize=4, label='Train Loss (BCE)', color='#06A77D')
    ax.plot(epochs_2, val_loss_2, 's-', linewidth=2, markersize=4, label='Validation Loss (BCE)', color='#D62828')
    
    # Mark best epoch
    ax.scatter([best_epoch_num_2], [best_val_loss_2], s=200, color='gold', marker='*',
              zorder=5, label=f'Best (Epoch {best_epoch_num_2}, Val Loss={best_val_loss_2:.5f})', edgecolor='black', linewidth=1.5)
    ax.axvline(best_epoch_num_2, color='gold', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (Weighted BCE)', fontsize=12, fontweight='bold')
    ax.set_title('Model 2: LSTM-GN Route Classifier\n(150 Epochs, Weighted BCE Loss, F1 Score Tracking)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add statistics for model 2
    train_final_2 = train_loss_2[-1]
    val_final_2 = val_loss_2[-1]
    best_f1 = train_f1[best_epoch_2]
    final_f1 = train_f1[-1]
    
    stats_text_2 = (
        f'Final Train Loss: {train_final_2:.6f}\n'
        f'Final Val Loss: {val_final_2:.6f}\n'
        f'Best F1 Score: {best_f1:.4f}\n'
        f'Final F1 Score: {final_f1:.4f}\n\n'
        f'Overfitting Analysis:\n'
        f'Gap at Best: {best_val_loss_2 - train_loss_2[best_epoch_2]:.6f}\n'
        f'Gap at End: {val_final_2 - train_final_2:.6f}'
    )
    ax.text(0.98, 0.97, stats_text_2, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7, edgecolor='black', linewidth=1.5))
    
    plt.suptitle('ML Model Training Curves - Learning Progress & Overfitting Analysis', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {output_path}")
    
    # Print detailed analysis
    print("\n" + "="*90)
    print("TRAINING CURVE ANALYSIS")
    print("="*90)
    
    print("\n" + "─" * 90)
    print("MODEL 1: GraphSAGE Edge Travel Time Prediction (200 Epochs)")
    print("─" * 90)
    print(f"\nLoss Function: Log-MSE on travel time predictions")
    print(f"Optimizer: Adam with Cosine Learning Rate Schedule")
    print(f"\nInitial Performance:")
    print(f"   Train Loss: {train_loss[0]:.6f}")
    print(f"   Val Loss: {val_loss[0]:.6f}")
    print(f"   Generalization Gap: {val_loss[0] - train_loss[0]:.6f}")
    
    print(f"\nBest Performance (Epoch {best_epoch_num}):")
    print(f"   Train Loss: {train_loss[best_epoch]:.6f}")
    print(f"   Val Loss: {best_val_loss:.6f}")
    print(f"   Generalization Gap: {best_val_loss - train_loss[best_epoch]:.6f}")
    
    print(f"\nFinal Performance (Epoch 200):")
    print(f"   Train Loss: {train_final:.6f}")
    print(f"   Val Loss: {val_final:.6f}")
    print(f"   Generalization Gap: {val_final - train_final:.6f}")
    
    print(f"\nLearning Efficiency:")
    print(f"   Train Loss Improvement: {train_improved:.1f}% from epoch 1 to 200")
    print(f"   Val Loss Improvement: {val_improved:.1f}% from epoch 1 to epoch {best_epoch_num}")
    print(f"   Val Loss Stability: {abs((val_loss[-1] - val_loss[best_epoch]) / val_loss[best_epoch] * 100):.1f}% change from best to end")
    
    # Detect overfitting stage
    print(f"\n OVERFITTING ANALYSIS:")
    if best_val_loss - train_loss[best_epoch] < 0.01:
        print(f"   Minimal overfitting: Gap between train and val at best epoch is very small")
    elif best_val_loss - train_loss[best_epoch] < 0.05:
        print(f"   Mild overfitting: Small generalization gap - typical and acceptable")
    else:
        print(f"   Moderate overfitting: Noticeable gap between train and val curves")
    
    if val_final > val_loss[best_epoch] * 1.05:
        print(f"   Val loss divergence detected after epoch {best_epoch_num}")
        print(f"     (increases by {(val_final - val_loss[best_epoch]) / val_loss[best_epoch] * 100:.1f}% by epoch 200)")
    
    print("\n" + "─" * 90)
    print("MODEL 2: LSTM-GN Route Classifier (150 Epochs)")
    print("─" * 90)
    print(f"\nLoss Function: Weighted Binary Cross-Entropy (dynamic pos_weight)")
    print(f"Metric: F1-Score for route classification accuracy")
    print(f"Architecture: LSTM + Graph Neural Network with gated message passing")
    print(f"\nInitial Performance:")
    print(f"   Train Loss: {train_loss_2[0]:.6f}")
    print(f"   Val Loss: {val_loss_2[0]:.6f}")
    print(f"   Train F1: {train_f1[0]:.4f}")
    
    print(f"\nBest Performance (Epoch {best_epoch_num_2}):")
    print(f"   Train Loss: {train_loss_2[best_epoch_2]:.6f}")
    print(f"   Val Loss: {best_val_loss_2:.6f}")
    print(f"   F1 Score: {train_f1[best_epoch_2]:.4f}")
    print(f"   Generalization Gap: {best_val_loss_2 - train_loss_2[best_epoch_2]:.6f}")
    
    print(f"\nFinal Performance (Epoch 150):")
    print(f"   Train Loss: {train_final_2:.6f}")
    print(f"   Val Loss: {val_final_2:.6f}")
    print(f"   F1 Score: {final_f1:.4f}")
    
    print(f"\nLearning Efficiency:")
    print(f"   Train Loss Improvement: {(train_loss_2[0] - train_final_2) / train_loss_2[0] * 100:.1f}% from epoch 1 to 150")
    print(f"   Val Loss Improvement: {(val_loss_2[0] - best_val_loss_2) / val_loss_2[0] * 100:.1f}% from epoch 1 to epoch {best_epoch_num_2}")
    print(f"   F1 Score Improvement: {(final_f1 - train_f1[0]) / train_f1[0] * 100:.1f}%")
    
    
    plt.show()


if __name__ == '__main__':
    plot_training_curves()
