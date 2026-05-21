
"""
Analyze and visualize edge feature distributions (speed and jam factor) from a snapshot.
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def analyze_snapshot(snapshot_path: str):
    """
    Analyze edge feature distributions from a snapshot.
    
    Args:
        snapshot_path: Path to snapshot JSON file
    """
    with open(snapshot_path, 'r') as f:
        data = json.load(f)
    
    speeds = []
    jam_factors = []
    congestion_levels = []
    
    # Extract feature data - only edges with both speed and jam_factor
    for feature in data.get('features', []):
        properties = feature.get('properties', {})
        
        speed = properties.get('current_speed_kph')
        jam = properties.get('jam_factor')
        
        # Only include edges with both values
        if speed is not None and jam is not None:
            speeds.append(speed)
            jam_factors.append(jam)
            
            # Classify congestion level based on jam_factor
            if jam > 5:
                congestion_levels.append('Congested (jam > 5)')
            elif jam > 2:
                congestion_levels.append('Slow (2 < jam ≤ 5)')
            else:
                congestion_levels.append('Free Flow (jam ≤ 2)')
    
    if not speeds or not jam_factors:
        print("No valid edge data found in snapshot")
        return
    
    speeds_array = np.array(speeds)
    jam_factors_array = np.array(jam_factors)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Get metadata
    metadata = data.get('metadata', {})
    timestamp = metadata.get('timestamp', 'Unknown')
    city = metadata.get('city', 'Unknown')
    
    fig.suptitle(f'Edge Feature Distribution - {city} ({timestamp})\nTotal Edges: {len(speeds)}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Speed histogram
    ax1 = fig.add_subplot(gs[0, 0])
    counts, bins, patches = ax1.hist(speeds_array, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(speeds_array), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(speeds_array):.1f} kph')
    ax1.axvline(np.median(speeds_array), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(speeds_array):.1f} kph')
    ax1.set_xlabel('Current Speed (kph)', fontsize=11)
    ax1.set_ylabel('Number of Edges', fontsize=11)
    ax1.set_title('Distribution of Current Speed', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Speed statistics
    speed_stats = (
        f'Count: {len(speeds)}\n'
        f'Mean: {np.mean(speeds_array):.2f} kph\n'
        f'Median: {np.median(speeds_array):.2f} kph\n'
        f'Std Dev: {np.std(speeds_array):.2f} kph\n'
        f'Min: {np.min(speeds_array):.2f} kph\n'
        f'Max: {np.max(speeds_array):.2f} kph'
    )
    ax1.text(0.98, 0.97, speed_stats, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 2. Jam factor histogram
    ax2 = fig.add_subplot(gs[0, 1])
    counts, bins, patches = ax2.hist(jam_factors_array, bins=40, edgecolor='black', alpha=0.7, color='salmon')
    ax2.axvline(np.mean(jam_factors_array), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(jam_factors_array):.2f}')
    ax2.axvline(np.median(jam_factors_array), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(jam_factors_array):.2f}')
    ax2.axvline(2, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Threshold (2)')
    ax2.axvline(5, color='darkred', linestyle=':', linewidth=2, alpha=0.7, label='Threshold (5)')
    ax2.set_xlabel('Jam Factor', fontsize=11)
    ax2.set_ylabel('Number of Edges', fontsize=11)
    ax2.set_title('Distribution of Jam Factor', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Jam factor statistics
    jam_stats = (
        f'Count: {len(jam_factors)}\n'
        f'Mean: {np.mean(jam_factors_array):.3f}\n'
        f'Median: {np.median(jam_factors_array):.3f}\n'
        f'Std Dev: {np.std(jam_factors_array):.3f}\n'
        f'Min: {np.min(jam_factors_array):.3f}\n'
        f'Max: {np.max(jam_factors_array):.3f}'
    )
    ax2.text(0.98, 0.97, jam_stats, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
    
    # 3. Congestion level pie chart
    ax3 = fig.add_subplot(gs[1, 0])
    congestion_counts = pd.Series(congestion_levels).value_counts()
    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
    wedges, texts, autotexts = ax3.pie(congestion_counts.values, 
                                         labels=congestion_counts.index,
                                         autopct='%1.1f%%',
                                         colors=colors,
                                         startangle=90,
                                         textprops={'fontsize': 10})
    ax3.set_title('Network Congestion Status', fontsize=12, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # 4. Speed distribution by congestion level
    ax4 = fig.add_subplot(gs[1, 1])
    congested_mask = jam_factors_array > 5
    slow_mask = (jam_factors_array > 2) & (jam_factors_array <= 5)
    free_mask = jam_factors_array <= 2
    
    speed_by_congestion = [
        speeds_array[free_mask],
        speeds_array[slow_mask],
        speeds_array[congested_mask]
    ]
    
    bp = ax4.boxplot(speed_by_congestion, 
                      labels=['Free Flow\n(jam ≤ 2)', 'Slow\n(2 < jam ≤ 5)', 'Congested\n(jam > 5)'],
                      patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['#2ca02c', '#ff7f0e', '#d62728']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Current Speed (kph)', fontsize=11)
    ax4.set_title('Speed Distribution by Congestion Level', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Scatter plot: Jam Factor vs Speed
    ax5 = fig.add_subplot(gs[2, :])
    scatter = ax5.scatter(speeds_array, jam_factors_array, alpha=0.5, s=10, c=jam_factors_array, 
                         cmap='RdYlGn_r', edgecolors='none')
    ax5.axhline(2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Slow threshold (jam = 2)')
    ax5.axhline(5, color='darkred', linestyle=':', linewidth=1.5, alpha=0.7, label='Congested threshold (jam = 5)')
    ax5.set_xlabel('Current Speed (kph)', fontsize=11)
    ax5.set_ylabel('Jam Factor', fontsize=11)
    ax5.set_title('Relationship between Speed and Jam Factor', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Jam Factor', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(snapshot_path).stem + '_edge_features.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Edge Feature Distribution Analysis - {city} ({timestamp})")
    print(f"{'='*80}")
    print(f"\nTotal Edges: {len(speeds)}")
    print(f"\nCURRENT SPEED (kph):")
    print(f"  Mean: {np.mean(speeds_array):.2f}")
    print(f"  Median: {np.median(speeds_array):.2f}")
    print(f"  Std Dev: {np.std(speeds_array):.2f}")
    print(f"  Min: {np.min(speeds_array):.2f}")
    print(f"  Max: {np.max(speeds_array):.2f}")
    print(f"  Q1 (25%): {np.percentile(speeds_array, 25):.2f}")
    print(f"  Q3 (75%): {np.percentile(speeds_array, 75):.2f}")
    
    print(f"\nJAM FACTOR:")
    print(f"  Mean: {np.mean(jam_factors_array):.3f}")
    print(f"  Median: {np.median(jam_factors_array):.3f}")
    print(f"  Std Dev: {np.std(jam_factors_array):.3f}")
    print(f"  Min: {np.min(jam_factors_array):.3f}")
    print(f"  Max: {np.max(jam_factors_array):.3f}")
    print(f"  Q1 (25%): {np.percentile(jam_factors_array, 25):.3f}")
    print(f"  Q3 (75%): {np.percentile(jam_factors_array, 75):.3f}")
    
    print(f"\nCONGESTION STATUS:")
    for level, count in congestion_counts.items():
        pct = (count / len(jam_factors)) * 100
        print(f"  {level}: {count} edges ({pct:.1f}%)")
    
    print(f"\nAVERAGE SPEED BY CONGESTION LEVEL:")
    print(f"  Free Flow (jam ≤ 2): {np.mean(speeds_array[free_mask]):.2f} kph ({np.sum(free_mask)} edges)")
    print(f"  Slow (2 < jam ≤ 5): {np.mean(speeds_array[slow_mask]):.2f} kph ({np.sum(slow_mask)} edges)")
    print(f"  Congested (jam > 5): {np.mean(speeds_array[congested_mask]):.2f} kph ({np.sum(congested_mask)} edges)")
    
    print(f"{'='*80}\n")
    
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Use the snapshot
        snapshot_path = r'c:\Users\Nojus\Desktop\bakalauras\backend\cities\vilnius\snapshots\2026-04-01T16-10-23.255138.json'
        print(f"Using default snapshot: {snapshot_path}")
    else:
        snapshot_path = sys.argv[1]
    
    analyze_snapshot(snapshot_path)
