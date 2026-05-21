
"""
Generate a histogram of travel times across all edges from a snapshot.
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def calculate_travel_time(length_m: float, speed_kph: float) -> float:
    """
    Calculate travel time in seconds.
    """
    if speed_kph is None or speed_kph <= 0 or length_m is None or length_m <= 0:
        return None
    # Convert speed from kph to m/s: kph * 1000 / 3600 = kph * 5/18
    speed_ms = speed_kph * 1000 / 3600
    return length_m / speed_ms


def analyze_snapshot(snapshot_path: str):
    """
    Analyze travel times from a snapshot file and create a histogram.
    """
    with open(snapshot_path, 'r') as f:
        data = json.load(f)
    
    travel_times = []
    
    for feature in data.get('features', []):
        properties = feature.get('properties', {})
        
        # Skip features without required properties
        if 'length' not in properties or 'current_speed_kph' not in properties:
            continue
        
        length = properties['length']
        speed = properties['current_speed_kph']
        
        travel_time = calculate_travel_time(length, speed)
        if travel_time is not None:
            travel_times.append(travel_time)
    
    if not travel_times:
        print("No valid edges found in snapshot")
        return
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    
    travel_times_array = np.array(travel_times)
    
    # Statistics
    mean_time = np.mean(travel_times_array)
    median_time = np.median(travel_times_array)
    std_time = np.std(travel_times_array)
    min_time = np.min(travel_times_array)
    max_time = np.max(travel_times_array)
    
    # Create histogram
    n, bins, patches = ax.hist(travel_times_array, bins=50, edgecolor='black', alpha=0.7)
    
    # Add vertical lines for mean and median
    ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.2f}s')
    ax.axvline(median_time, color='green', linestyle='--', linewidth=2, label=f'Median: {median_time:.2f}s')
    
    # Labels and title
    metadata = data.get('metadata', {})
    timestamp = metadata.get('timestamp', 'Unknown')
    city = metadata.get('city', 'Unknown')
    
    ax.set_xlabel('Travel Time (seconds)', fontsize=12)
    ax.set_ylabel('Number of Edges', fontsize=12)
    ax.set_title(f'Travel Time Distribution - {city} ({timestamp})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = (
        f'Count: {len(travel_times)}\n'
        f'Mean: {mean_time:.2f}s\n'
        f'Median: {median_time:.2f}s\n'
        f'Std Dev: {std_time:.2f}s\n'
        f'Min: {min_time:.2f}s\n'
        f'Max: {max_time:.2f}s'
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(snapshot_path).stem + '_travel_times.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Histogram saved to: {output_path}")
    
    # Print statistics
    print(f"\nTravel Time Statistics for {city} ({timestamp}):")
    print(f"  Total edges: {len(travel_times)}")
    print(f"  Mean: {mean_time:.2f} seconds")
    print(f"  Median: {median_time:.2f} seconds")
    print(f"  Std Dev: {std_time:.2f} seconds")
    print(f"  Min: {min_time:.2f} seconds")
    print(f"  Max: {max_time:.2f} seconds")
    print(f"  25th percentile: {np.percentile(travel_times_array, 25):.2f} seconds")
    print(f"  75th percentile: {np.percentile(travel_times_array, 75):.2f} seconds")
    
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Use the snapshot
        snapshot_path = r'c:\Users\Nojus\Desktop\bakalauras\backend\cities\vilnius\snapshots\2026-04-01T16-10-23.255138.json'
        print(f"Using default snapshot: {snapshot_path}")
    else:
        snapshot_path = sys.argv[1]
    
    analyze_snapshot(snapshot_path)
