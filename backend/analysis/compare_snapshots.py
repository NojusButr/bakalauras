
"""
Compare edge statistics across multiple snapshots to show temporal variation.
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def analyze_snapshot(snapshot_path: str) -> dict:
    """
    Analyze a single snapshot file.
    
    Args:
        snapshot_path: Path to snapshot JSON file
    
    Returns:
        Dict with analysis results
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
        
        speed = properties.get('current_speed_kph')
        jam_factor = properties.get('jam_factor')
        
        # Only include edges with both values
        if speed is not None and jam_factor is not None:
            speeds.append(speed)
            jam_factors.append(jam_factor)
            total_count += 1
            
            if jam_factor > 5:
                congested_count += 1
    
    metadata = data.get('metadata', {})
    
    return {
        'timestamp': metadata.get('timestamp', ''),
        'avg_speed': np.mean(speeds) if speeds else 0,
        'avg_jam_factor': np.mean(jam_factors) if jam_factors else 0,
        'congested_percentage': (congested_count / total_count * 100) if total_count > 0 else 0,
        'edge_count': total_count
    }


def extract_datetime(filename: str) -> datetime:
    """
    Extract datetime from snapshot filename.
    
    Args:
        filename: Snapshot filename (format: YYYY-MM-DDTHH-MM-SS.mmmmmm.json)
    
    Returns:
        datetime object
    """
    try:
        # Format: 2026-04-01T16-10-23.255138.json
        name = filename.replace('.json', '').split('.')[0]  # Remove extension and microseconds
        # name is now: 2026-04-01T16-10-23
        parts = name.split('T')
        date_part = parts[0]  # 2026-04-01
        time_part = parts[1]  # 16-10-23
        
        # Convert time dashes to colons
        time_formatted = time_part.replace('-', ':')  # 16:10:23
        
        # Combine and parse
        datetime_str = f"{date_part} {time_formatted}"  # 2026-04-01 16:10:23
        return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return datetime.now()


def main():
    # Get snapshots directory
    script_dir = Path(__file__).parent.parent
    snapshots_dir = script_dir / 'cities' / 'vilnius' / 'snapshots'
    
    if not snapshots_dir.exists():
        print(f"Snapshots directory not found: {snapshots_dir}")
        return
    
    # Get all snapshot files sorted by time
    snapshot_files = sorted(snapshots_dir.glob('*.json'))
    print(f"Found {len(snapshot_files)} total snapshots\n")
    
    # Select representative snapshots from different times
    # Strategy: pick snapshots spread across dates and times
    selected_indices = []
    
    if len(snapshot_files) >= 4:
        # Pick first, roughly 1/3 mark, 2/3 mark, and last
        selected_indices = [0, len(snapshot_files)//3, 2*len(snapshot_files)//3, -1]
    elif len(snapshot_files) >= 3:
        selected_indices = [0, len(snapshot_files)//2, -1]
    else:
        selected_indices = list(range(len(snapshot_files)))
    
    # Analyze selected snapshots
    results = []
    print("Analyzing selected snapshots:")
    print("-" * 80)
    
    for idx in selected_indices:
        snapshot_file = snapshot_files[idx]
        print(f"Analyzing {snapshot_file.name}...", end='', flush=True)
        
        analysis = analyze_snapshot(str(snapshot_file))
        
        # Parse datetime for nice display
        dt = extract_datetime(snapshot_file.name)
        timestamp_display = dt.strftime('%b %d %H:%M')
        
        results.append({
            'Snapshot': snapshot_file.name,
            'DateTime': timestamp_display,
            'Avg Speed (kph)': round(analysis['avg_speed'], 2),
            'Avg Jam Factor': round(analysis['avg_jam_factor'], 3),
            'Congested Edges (%)': round(analysis['congested_percentage'], 1),
            'Total Edges': analysis['edge_count']
        })
        print(" done")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Display as formatted table
    print("\n" + "="*100)
    print("SNAPSHOT COMPARISON - TEMPORAL VARIATION")
    print("="*100)
    
    # Print formatted table with nice descriptions
    for idx, row in df.iterrows():
        print(f"\n📍 Snapshot from {row['DateTime']}")
        print(f"   File: {row['Snapshot']}")
        print(f"    Average Speed: {row['Avg Speed (kph)']} kph")
        print(f"    Average Jam Factor: {row['Avg Jam Factor']}")
        print(f"    Congested Edges (jam > 5): {row['Congested Edges (%)']}%")
        print(f"    Total Edges Analyzed: {row['Total Edges']}")
    
    print("\n" + "="*100)
    print("\nComparison Summary Table:")
    print("="*100)
    
    # Create a cleaner display table
    display_df = df[['DateTime', 'Avg Speed (kph)', 'Avg Jam Factor', 'Congested Edges (%)']].copy()
    print(display_df.to_string(index=False))
    print("="*100)
    
    # Calculate variation metrics
    print("\nVariation Across Selected Snapshots:")
    print("-" * 100)
    print(f"Speed Range: {df['Avg Speed (kph)'].min():.2f} - {df['Avg Speed (kph)'].max():.2f} kph")
    print(f"Speed Variation: ±{(df['Avg Speed (kph)'].std()):.2f} kph (std dev)")
    print(f"Congestion Range: {df['Congested Edges (%)'].min():.1f}% - {df['Congested Edges (%)'].max():.1f}%")
    print(f"Congestion Variation: ±{(df['Congested Edges (%)'].std()):.1f}% (std dev)")
    print("-" * 100)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Network Conditions Across Selected Snapshots', fontsize=14, fontweight='bold')
    
    x_pos = np.arange(len(df))
    
    # 1. Average Speed comparison
    ax = axes[0]
    bars = ax.bar(x_pos, df['Avg Speed (kph)'], color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_ylabel('Average Speed (kph)', fontsize=11, fontweight='bold')
    ax.set_title('Average Speed Variation', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['DateTime'], rotation=45, ha='right')
    ax.set_ylim(0, max(df['Avg Speed (kph)']) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df['Avg Speed (kph)'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. Jam Factor comparison
    ax = axes[1]
    bars = ax.bar(x_pos, df['Avg Jam Factor'], color='salmon', edgecolor='black', alpha=0.7)
    ax.set_ylabel('Average Jam Factor', fontsize=11, fontweight='bold')
    ax.set_title('Jam Factor Variation', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['DateTime'], rotation=45, ha='right')
    ax.set_ylim(0, max(df['Avg Jam Factor']) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df['Avg Jam Factor'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
               f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. Congestion percentage comparison
    ax = axes[2]
    bars = ax.bar(x_pos, df['Congested Edges (%)'], color='lightcoral', edgecolor='black', alpha=0.7)
    ax.set_ylabel('Congested Edges (%)', fontsize=11, fontweight='bold')
    ax.set_title('Congestion Level Variation', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['DateTime'], rotation=45, ha='right')
    ax.set_ylim(0, max(df['Congested Edges (%)']) * 1.3)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df['Congested Edges (%)'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'snapshot_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Save CSV
    csv_path = 'snapshot_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")
    
    plt.show()


if __name__ == '__main__':
    main()
