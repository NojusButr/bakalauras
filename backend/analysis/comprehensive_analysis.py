"""
Comprehensive analysis of GNN performance experiments.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_experiment(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def get_latest_experiment():
    """Find and return the latest degraded_full_*.json file."""
    script_dir = Path(__file__).parent.parent
    experiments_dir = script_dir / "experiments"
    experiment_files = sorted(experiments_dir.glob("degraded_full_*.json"))
    if not experiment_files:
        raise FileNotFoundError("No experiment files found in experiments/ directory")
    return load_experiment(experiment_files[-1])


def analyze_main_results(exp):
    """Generate main results table."""
    summary = exp["summary_by_level"]
    original_order = list(exp["config"]["data_levels"])
    data_levels = [level for level in original_order if level != "minor"] + ["minor"]
    
    print("\n" + "="*130)
    print("TABLE 1: MAIN RESULTS - GNN vs TRAFFIC-AWARE DIJKSTRA")
    print("="*130)
    print(f"{'Level':<8} {'GNN Wins':<12} {'Traffic Wins':<14} {'Ties':<8} {'Avg Duration':<16} {'Time Diff':<14} {'Improvement':<14}")
    print(f"{'':8} {'':12} {'':14} {'':8} {'(minutes)':<16} {'(seconds)':<14} {'(%)':<14}")
    print("-"*130)
    
    for level in data_levels:
        stats = summary[level]
        ties = stats['total'] - stats['gnn_wins'] - stats['traffic_wins']
        avg_duration = stats.get('avg_route_duration_minutes', stats['avg_traffic_time_s'] / 60)
        time_diff = stats['avg_time_diff_s']
        improvement = stats.get('relative_improvement_pct', (time_diff / stats['avg_gnn_time_s'] * 100) if stats['avg_gnn_time_s'] > 0 else 0)
        
        print(f"{str(level):<8} {stats['gnn_wins']:<12} {stats['traffic_wins']:<14} {ties:<8} "
              f"{avg_duration:>14.2f}  {time_diff:>+12.2f}  {improvement:>+12.2f}")
    
    print("="*130)


def analyze_computation_times(exp):
    """Generate computation time analysis."""
    summary = exp["summary_by_level"]
    original_order = list(exp["config"]["data_levels"])
    data_levels = [level for level in original_order if level != "minor"] + ["minor"]
    
    print("\n" + "="*100)
    print("TABLE 2: AVERAGE COMPUTATION TIME PER METHOD (across all degradation levels)")
    print("="*100)
    
    methods = ['shortest', 'traffic', 'gnn', 'classifier']
    all_times = {method: [] for method in methods}
    
    for level in data_levels:
        stats = summary[level]
        all_times['shortest'].append(stats.get('avg_compute_shortest_s', 0))
        all_times['traffic'].append(stats.get('avg_compute_traffic_s', 0))
        all_times['gnn'].append(stats.get('avg_compute_gnn_s', 0))
        all_times['classifier'].append(stats.get('avg_compute_classifier_s', 0))
    
    print(f"{'Method':<15} {'Avg Time (ms)':<20} {'Min (ms)':<20} {'Max (ms)':<20}")
    print("-"*100)
    
    for method in methods:
        times_ms = [t * 1000 for t in all_times[method]]
        avg = np.mean(times_ms)
        min_t = np.min(times_ms)
        max_t = np.max(times_ms)
        print(f"{method:<15} {avg:>18.3f}  {min_t:>18.3f}  {max_t:>18.3f}")
    
    print("="*100)
    
    return all_times, data_levels


def analyze_by_distance(exp):
    """Analyze GNN win rate by route distance."""
    results = exp["results"]
    summary = exp["summary_by_level"]
    original_order = list(exp["config"]["data_levels"])
    data_levels = [level for level in original_order if level != "minor"] + ["minor"]
    
    # Define distance categories
    distance_categories = {
        'Short (<3km)': (0, 3),
        'Medium (3-7km)': (3, 7),
        'Long (>7km)': (7, float('inf'))
    }
    
    # Group routes by distance
    routes_by_distance = {cat: [] for cat in distance_categories}
    for route in results:
        dist = route.get('route_distance_km', 0)
        for cat, (min_d, max_d) in distance_categories.items():
            if min_d <= dist < max_d:
                routes_by_distance[cat].append(route)
                break
    
    print("\n" + "="*120)
    print("TABLE 3: GNN WIN RATE BY ROUTE DISTANCE CATEGORY")
    print("="*120)
    
    # Create header with data levels
    header = f"{'Distance Category':<20}"
    for level in data_levels:
        header += f" {str(level):<12}"
    print(header)
    print("-"*120)
    
    for cat in ['Short (<3km)', 'Medium (3-7km)', 'Long (>7km)']:
        routes = routes_by_distance[cat]
        if not routes:
            print(f"{cat:<20} (no routes)")
            continue
        
        row = f"{cat:<20}"
        for level in data_levels:
            level_routes = [r for r in routes if str(r['data_level']) == str(level)]
            if level_routes:
                gnn_wins = sum(1 for r in level_routes if str(r['winner']).lower() == 'gnn')
                win_rate = (gnn_wins / len(level_routes) * 100) if level_routes else 0
                row += f" {win_rate:>10.1f}%"
            else:
                row += f" {'N/A':>10}"
        print(row)
    
    print("="*120)
    
    return routes_by_distance, distance_categories


def analyze_crisis_vs_normal(exp):
    """Compare GNN win rates for crisis vs normal scenarios."""
    results = exp["results"]
    summary = exp["summary_by_level"]
    original_order = list(exp["config"]["data_levels"])
    data_levels = [level for level in original_order if level != "minor"] + ["minor"]
    
    print("\n" + "="*110)
    print("TABLE 4: GNN WIN RATE - CRISIS vs NORMAL SCENARIOS")
    print("="*110)
    print(f"{'Data Level':<15} {'Crisis Wins':<20} {'Normal Wins':<20} {'Crisis Rate (%)':<20} {'Normal Rate (%)':<20}")
    print("-"*110)
    
    for level in data_levels:
        level_results = [r for r in results if str(r['data_level']) == str(level)]
        
        crisis = [r for r in level_results if r['is_crisis']]
        normal = [r for r in level_results if not r['is_crisis']]
        
        crisis_gnn_wins = sum(1 for r in crisis if str(r['winner']).lower() == 'gnn')
        normal_gnn_wins = sum(1 for r in normal if str(r['winner']).lower() == 'gnn')
        
        crisis_rate = (crisis_gnn_wins / len(crisis) * 100) if crisis else 0
        normal_rate = (normal_gnn_wins / len(normal) * 100) if normal else 0
        
        print(f"{str(level):<15} {crisis_gnn_wins:<20} {normal_gnn_wins:<20} {crisis_rate:>18.1f}  {normal_rate:>18.1f}")
    
    print("="*110)


def analyze_shortest_and_classifier(exp):
    """Report average travel times for shortest path and classifier."""
    results = exp["results"]
    summary = exp["summary_by_level"]
    original_order = list(exp["config"]["data_levels"])
    data_levels = [level for level in original_order if level != "minor"] + ["minor"]
    
    print("\n" + "="*110)
    print("TABLE 5: SHORTEST PATH & CLASSIFIER - AVERAGE TRAVEL TIMES")
    print("="*110)
    print(f"{'Data Level':<15} {'Shortest (s)':<20} {'Classifier (s)':<20} {'vs Traffic':<20} {'vs GNN':<20}")
    print(f"{'':15} {'':20} {'':20} {'Classifier Diff':<20} {'Classifier Diff':<20}")
    print("-"*110)
    
    for level in data_levels:
        level_results = [r for r in results if str(r['data_level']) == str(level)]
        
        if not level_results:
            continue
        
        shortest_travel = np.mean([r['time_shortest_s'] for r in level_results])
        classifier_travel = np.mean([r['time_classifier_s'] for r in level_results])
        traffic_time = summary[level]['avg_traffic_time_s']
        gnn_time = summary[level]['avg_gnn_time_s']
        
        classifier_vs_traffic = classifier_travel - traffic_time
        classifier_vs_gnn = classifier_travel - gnn_time
        
        print(f"{str(level):<15} {shortest_travel:>18.2f}  {classifier_travel:>18.2f}  "
              f"{classifier_vs_traffic:>+18.2f}  {classifier_vs_gnn:>+18.2f}")
    
    print("="*110)


def compute_missing_stats(exp):
    """Compute missing statistics from results array."""
    results = exp["results"]
    summary = exp["summary_by_level"]
    original_order = list(exp["config"]["data_levels"])
    data_levels = [level for level in original_order if level != "minor"] + ["minor"]
    
    for level in data_levels:
        level_results = [r for r in results if str(r['data_level']) == str(level)]
        
        if level_results:
            summary[level]['avg_compute_shortest_s'] = np.mean([r['compute_time_shortest_s'] for r in level_results])
            summary[level]['avg_compute_classifier_s'] = np.mean([r['compute_time_classifier_s'] for r in level_results])
            summary[level]['avg_route_duration_minutes'] = np.mean([r['avg_route_duration_minutes'] for r in level_results])
            
            # Count shortest and classifier wins
            shortest_wins = sum(1 for r in level_results if str(r['winner']).lower() == 'shortest')
            classifier_wins = sum(1 for r in level_results if str(r['winner']).lower() == 'classifier')
            summary[level]['shortest_wins'] = shortest_wins
            summary[level]['classifier_wins'] = classifier_wins


def create_visualizations(exp):
    """Create publication-ready charts."""
    summary = exp["summary_by_level"]
    results = exp["results"]
    original_order = list(exp["config"]["data_levels"])
    data_levels = [level for level in original_order if level != "minor"] + ["minor"]
    
    # Prepare data
    gnn_wins = [summary[level]['gnn_wins'] for level in data_levels]
    traffic_wins = [summary[level]['traffic_wins'] for level in data_levels]
    ties = [summary[level]['total'] - summary[level]['gnn_wins'] - summary[level]['traffic_wins'] 
            for level in data_levels]
    win_rates = [(summary[level]['gnn_wins'] / summary[level]['total'] * 100) for level in data_levels]
    
    # Chart 1: Win rate vs data availability (line chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(data_levels)), win_rates, 'o-', linewidth=2.5, markersize=8, 
            color='#2E86AB', markerfacecolor='#A23B72', markeredgewidth=2, markeredgecolor='#2E86AB')
    ax.set_xlabel('Data Availability Level (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('GNN Win Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('GNN Win Rate vs Data Availability', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(data_levels)))
    ax.set_xticklabels(data_levels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(win_rates) * 1.15)
    
    for i, rate in enumerate(win_rates):
        ax.text(i, rate + 1.5, f'{rate:.1f}%', ha='center', fontsize=9)
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig('chart_1_win_rate.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\n Chart 1 saved: chart_1_win_rate.png")
    
    # Chart 2: Stacked bar chart (GNN, Traffic, Ties)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(data_levels))
    width = 0.6
    
    bars1 = ax.bar(x, gnn_wins, width, label='GNN Wins', color='#2E86AB', alpha=0.9)
    bars2 = ax.bar(x, traffic_wins, width, bottom=gnn_wins, label='Traffic Wins', color='#A23B72', alpha=0.9)
    bars3 = ax.bar(x, ties, width, bottom=np.array(gnn_wins) + np.array(traffic_wins), 
                   label='Ties', color='#F18F01', alpha=0.9)
    
    ax.set_xlabel('Data Availability Level (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Routes', fontsize=11, fontweight='bold')
    ax.set_title('Route Outcomes by Data Availability Level', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(data_levels, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig('chart_2_stacked_outcomes.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(" Chart 2 saved: chart_2_stacked_outcomes.png")
    
    # Chart 3: Average computation time per method
    methods = ['shortest', 'traffic', 'gnn', 'classifier']
    method_labels = ['Shortest', 'Traffic\nDijkstra', 'GNN', 'Classifier']
    avg_times = []
    
    for method in methods:
        key = f'avg_compute_{method}_s'
        times = [summary[level].get(key, 0) for level in data_levels]
        avg_times.append(np.mean(times) * 1000)  # Convert to ms
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = ax.bar(method_labels, avg_times, color=colors, alpha=0.9, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Average Computation Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Average Computation Time by Method', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bar, time in zip(bars, avg_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig('chart_3_computation_time.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(" Chart 3 saved: chart_3_computation_time.png")
    
    # Chart 4: GNN win rate by route distance at selected levels
    distance_categories = {
        'Short (<3km)': (0, 3),
        'Medium (3-7km)': (3, 7),
        'Long (>7km)': (7, float('inf'))
    }
    
    routes_by_distance = {cat: [] for cat in distance_categories}
    for route in results:
        dist = route.get('route_distance_km', 0)
        for cat, (min_d, max_d) in distance_categories.items():
            if min_d <= dist < max_d:
                routes_by_distance[cat].append(route)
                break
    
    # Select degradation levels for display (spread across the range)
    selected_levels = [90, 70, 50, 30, 10, 0, 'minor']
    
    distance_cats = ['Short (<3km)', 'Medium (3-7km)', 'Long (>7km)']
    x = np.arange(len(selected_levels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors_dist = ['#2E86AB', '#A23B72', '#F18F01']
    for i, cat in enumerate(distance_cats):
        win_rates_cat = []
        for level in selected_levels:
            level_routes = [r for r in routes_by_distance[cat] if str(r['data_level']) == str(level)]
            if level_routes:
                gnn_w = sum(1 for r in level_routes if str(r['winner']).lower() == 'gnn')
                rate = (gnn_w / len(level_routes) * 100)
            else:
                rate = 0
            win_rates_cat.append(rate)
        
        ax.bar(x + i*width, win_rates_cat, width, label=cat, color=colors_dist[i], alpha=0.9, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Data Availability Level (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('GNN Win Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('GNN Win Rate by Route Distance Category', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(selected_levels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 100)
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig('chart_4_by_distance.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(" Chart 4 saved: chart_4_by_distance.png")


def main():
    """Run comprehensive analysis."""
    print("Loading experiment data...")
    exp = get_latest_experiment()
    
    # Compute missing statistics
    compute_missing_stats(exp)
    
    # Generate all analyses
    analyze_main_results(exp)
    analyze_computation_times(exp)
    analyze_by_distance(exp)
    analyze_crisis_vs_normal(exp)
    analyze_shortest_and_classifier(exp)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(exp)
    
    print("\n" + "="*130)
    print("ANALYSIS COMPLETE!")
    print("="*130)


if __name__ == "__main__":
    main()
