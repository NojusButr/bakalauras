
"""
Analyze GNN performance vs data availability from degraded network experiments.
Generates: plot of win rate, and detailed results table.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_experiment(filepath):
    """Load experiment JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    # Find the latest degraded_full_*.json file
    script_dir = Path(__file__).parent.parent
    experiments_dir = script_dir / "experiments"
    experiment_files = sorted(experiments_dir.glob("degraded_full_*.json"))
    
    if not experiment_files:
        print("Error: No experiment files found in experiments/ directory")
        return
    
    experiment_file = experiment_files[-1] 
    print(f"Loading {experiment_file}...")
    exp = load_experiment(experiment_file)
    
    summary = exp["summary_by_level"]
    
    # Reorder data levels so "minor" is at the end
    original_order = list(exp["config"]["data_levels"])
    reordered = [level for level in original_order if level != "minor"] + ["minor"]
    
    data_levels = reordered
    
    # Calculate win rates and prepare data
    win_rates = []
    for level in data_levels:
        stats = summary[level]
        # Calculate gnn_win_rate if not present
        if 'gnn_win_rate' in stats:
            win_rate = stats['gnn_win_rate']
        else:
            win_rate = (stats['gnn_wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
        win_rates.append(win_rate)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== PLOT 1: Win Rate vs Data Availability =====
    ax1.plot(range(len(data_levels)), win_rates, 'o-', linewidth=2, markersize=8, 
             color='#2E86AB', markerfacecolor='#A23B72', markeredgewidth=2, markeredgecolor='#2E86AB')
    ax1.set_xlabel('Data Availability Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('GNN Win Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('GNN Win Rate vs Data Availability\n(Degraded Network Scenarios)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(data_levels)))
    ax1.set_xticklabels(data_levels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(win_rates) * 1.1)
    
    # Add value labels on points
    for i, rate in enumerate(win_rates):
        ax1.text(i, rate + 1.5, f'{rate:.1f}%', ha='center', fontsize=9)
    
    # ===== TABLE: Results Summary =====
    print("\n" + "="*90)
    print("EXPERIMENT SUMMARY: GNN vs Traffic-Aware Dijkstra")
    print("="*90)
    print(f"{'Data Level':<15} {'Routes':<10} {'GNN Wins':<12} {'Dijkstra':<12} {'Avg Time Diff':<15}")
    print(f"{'':15} {'Tested':<10} {'':12} {'Wins':<12} {'(minutes)':<15}")
    print("-" * 90)
    
    table_data = []
    for level in data_levels:
        stats = summary[level]
        # Calculate values if not present in new format
        if 'gnn_win_rate' not in stats:
            gnn_win_rate = (stats['gnn_wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
        else:
            gnn_win_rate = stats['gnn_win_rate']
            
        if 'avg_advantage_min' not in stats:
            # Convert avg_time_diff_s to minutes (assuming positive means GNN is better)
            avg_advantage_min = stats.get('avg_time_diff_s', 0) / 60
        else:
            avg_advantage_min = stats['avg_advantage_min']
        
        table_data.append({
            'level': level,
            'routes': stats['total'],
            'gnn_wins': stats['gnn_wins'],
            'dijkstra_wins': stats['traffic_wins'],
            'avg_time_diff': avg_advantage_min,
            'gnn_win_rate': gnn_win_rate
        })
        
        print(f"{level:<15} {stats['total']:<10} {stats['gnn_wins']:<12} "
              f"{stats['traffic_wins']:<12} {avg_advantage_min:>+.2f}{'':10}")
    
    print("="*90)
    
    # ===== PLOT 2: Wins Comparison =====
    x = np.arange(len(data_levels))
    width = 0.35
    
    gnn_wins = [summary[level]["gnn_wins"] for level in data_levels]
    dijkstra_wins = [summary[level]["traffic_wins"] for level in data_levels]
    
    bars1 = ax2.bar(x - width/2, gnn_wins, width, label='GNN Wins', color='#2E86AB', alpha=0.8)
    bars2 = ax2.bar(x + width/2, dijkstra_wins, width, label='Dijkstra Wins', color='#A23B72', alpha=0.8)
    
    ax2.set_xlabel('Data Availability Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Wins', fontsize=12, fontweight='bold')
    ax2.set_title('GNN vs Dijkstra Wins by Data Level', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(data_levels, rotation=45, ha='right')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('gnn_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n Plot saved to gnn_analysis.png")
    plt.close()
    
    # Print key insights
    print("\nKEY INSIGHTS:")
    print("-" * 90)
    best_level = max(table_data, key=lambda x: x['gnn_wins'])
    worst_level = min(table_data, key=lambda x: x['gnn_wins'])
    
    print(f"   Highest GNN performance: {best_level['level']} with {best_level['gnn_wins']} wins "
          f"({best_level['gnn_win_rate']:.1f}%)")
    print(f"   Lowest GNN performance: {worst_level['level']} with {worst_level['gnn_wins']} wins "
          f"({worst_level['gnn_win_rate']:.1f}%)")
    
    avg_positive_diff = np.mean([t['avg_time_diff'] for t in table_data if t['avg_time_diff'] > 0])
    if not np.isnan(avg_positive_diff):
        print(f"   Avg time advantage when GNN wins: {avg_positive_diff:.2f} minutes")
    print()


if __name__ == "__main__":
    main()
