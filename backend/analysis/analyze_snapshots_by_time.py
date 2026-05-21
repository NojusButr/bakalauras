
"""
Analyze snapshot statistics grouped by time of day.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_time_group(hour: int) -> str:
    """
    Classify hour into time of day group.
    
    Args:
        hour: Hour of day (0-23)
    
    Returns:
        Time group label
    """
    if 6 <= hour < 10:
        return "Morning (6-10)"
    elif 10 <= hour < 15:
        return "Midday (10-15)"
    elif 15 <= hour < 19:
        return "Evening (15-19)"
    else:  
        return "Night (19-6)"


def extract_hour_from_filename(filename: str) -> int:
    """
    Extract hour from snapshot filename (format: YYYY-MM-DDTHH-MM-SS.mmmmmm.json).
    """
    # Format: 2026-04-01T16-10-23.255138.json
    parts = filename.replace('.json', '').split('T')
    if len(parts) == 2:
        time_parts = parts[1].split('-')
        return int(time_parts[0])
    return -1


def analyze_snapshot(snapshot_path: str) -> dict:
    """
    Analyze a single snapshot file.
    """
    with open(snapshot_path, 'r') as f:
        data = json.load(f)
    
    speeds = []
    jam_factors = []
    congested_count = 0
    total_count = 0
    
    # Extract data from all features
    for feature in data.get('features', []):
        properties = feature.get('properties', {})
        
        # Skip if missing required fields
        if 'current_speed_kph' not in properties or 'jam_factor' not in properties:
            continue
        
        speed = properties['current_speed_kph']
        jam_factor = properties['jam_factor']
        
        if speed is None or jam_factor is None:
            continue
        
        speeds.append(speed)
        jam_factors.append(jam_factor)
        total_count += 1
        
        if jam_factor > 5:
            congested_count += 1
    
    return {
        'avg_speed': np.mean(speeds) if speeds else 0,
        'avg_jam_factor': np.mean(jam_factors) if jam_factors else 0,
        'congested_percentage': (congested_count / total_count * 100) if total_count > 0 else 0,
        'edge_count': total_count
    }


def main():
    #  snapshots directory
    script_dir = Path(__file__).parent.parent
    snapshots_dir = script_dir / 'cities' / 'vilnius' / 'snapshots'
    
    if not snapshots_dir.exists():
        print(f"Snapshots directory not found: {snapshots_dir}")
        return
    
    time_groups = {
        "Morning (6-10)": [],
        "Midday (10-15)": [],
        "Evening (15-19)": [],
        "Night (19-6)": []
    }
    
    snapshot_files = sorted(snapshots_dir.glob('*.json'))
    print(f"Found {len(snapshot_files)} snapshots")
    
    for snapshot_file in snapshot_files:
        hour = extract_hour_from_filename(snapshot_file.name)
        if hour < 0:
            continue
        
        group = get_time_group(hour)
        print(f"Analyzing {snapshot_file.name} (hour {hour}, group: {group})...", end='', flush=True)
        
        analysis = analyze_snapshot(str(snapshot_file))
        time_groups[group].append(analysis)
        print(" done")
    
    # Aggregate statistics for each time group
    results = []
    for group_name in ["Morning (6-10)", "Midday (10-15)", "Evening (15-19)", "Night (19-6)"]:
        analyses = time_groups[group_name]
        
        if not analyses:
            continue
        
        avg_speed = np.mean([a['avg_speed'] for a in analyses])
        avg_jam_factor = np.mean([a['avg_jam_factor'] for a in analyses])
        congested_pct = np.mean([a['congested_percentage'] for a in analyses])
        
        results.append({
            'Time of Day': group_name,
            'Snapshot Count': len(analyses),
            'Avg Speed (kph)': round(avg_speed, 2),
            'Avg Jam Factor': round(avg_jam_factor, 2),
            'Congested Edges (%)': round(congested_pct, 1)
        })
    
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("Snapshot Statistics by Time of Day")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Snapshot Statistics by Time of Day', fontsize=16, fontweight='bold')
    
    time_labels = df['Time of Day'].tolist()
    
    # 1. Snapshot count
    ax = axes[0, 0]
    ax.bar(time_labels, df['Snapshot Count'], color='skyblue', edgecolor='black')
    ax.set_ylabel('Number of Snapshots', fontsize=11)
    ax.set_title('Snapshot Count by Time of Day', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(df['Snapshot Count']):
        ax.text(i, v + 0.1, str(int(v)), ha='center', va='bottom', fontweight='bold')
    
    # 2. Average speed
    ax = axes[0, 1]
    bars = ax.bar(time_labels, df['Avg Speed (kph)'], color='lightgreen', edgecolor='black')
    ax.set_ylabel('Average Speed (kph)', fontsize=11)
    ax.set_title('Average Speed by Time of Day', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, max(df['Avg Speed (kph)']) * 1.15)
    for i, v in enumerate(df['Avg Speed (kph)']):
        ax.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Average jam factor
    ax = axes[1, 0]
    bars = ax.bar(time_labels, df['Avg Jam Factor'], color='lightsalmon', edgecolor='black')
    ax.set_ylabel('Average Jam Factor', fontsize=11)
    ax.set_title('Average Jam Factor by Time of Day', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, max(df['Avg Jam Factor']) * 1.15)
    for i, v in enumerate(df['Avg Jam Factor']):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Congested edges percentage
    ax = axes[1, 1]
    bars = ax.bar(time_labels, df['Congested Edges (%)'], color='lightcoral', edgecolor='black')
    ax.set_ylabel('Congested Edges (%)', fontsize=11)
    ax.set_title('Percentage of Congested Edges (jam > 5)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, max(df['Congested Edges (%)']) * 1.15)
    for i, v in enumerate(df['Congested Edges (%)']):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'snapshots_by_time_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Save CSV
    csv_path = 'snapshots_by_time_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")
    
    plt.show()


if __name__ == '__main__':
    main()
