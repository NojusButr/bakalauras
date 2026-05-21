
"""
Analyze degraded_full_*.csv file and compute aggregated statistics by level.
"""

import csv
from pathlib import Path

def get_latest_csv():
    """Find the latest degraded_full_*.csv file."""
    script_dir = Path(__file__).parent.parent
    experiments_dir = script_dir / "experiments"
    csv_files = sorted(experiments_dir.glob("degraded_full_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in experiments/ directory")
    return csv_files[-1]

def parse_bool(value):
    """Parse boolean from CSV."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('true', '1', 'yes')

def parse_float(value):
    """Parse float, return None if empty or invalid."""
    if not value or value == 'None':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def main():
    csv_file = get_latest_csv()
    print(f"Loading {csv_file.name}...")
    
    # Load CSV
    rows = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    # Recalculate per level
    levels_order = ["90", "80", "70", "60", "50", "40", "30", "20", "10", "5", "0", "minor"]
    
    print("\n" + "="*130)
    print("LEVEL-BY-LEVEL ANALYSIS")
    print("="*130)
    print(f"{'Level':<8} {'GNN':<5} {'Traffic':<8} {'Ties':<6} {'Total':<6} {'Avg Diff':<12} {'Avg Duration':<14} {'Rel Imp':<10}")
    print(f"{'':8} {'Wins':<5} {'Wins':<8} {'':6} {'':6} {'(seconds)':<12} {'(minutes)':<14} {'(%)':<10}")
    print("-"*130)
    
    for level in levels_order:
        level_rows = [r for r in rows if str(r['data_level']) == level]
        
        # Only rows where both traffic and gnn have data
        valid = []
        for r in level_rows:
            time_traffic = parse_float(r['time_traffic_s'])
            time_gnn = parse_float(r['time_gnn_s'])
            if time_traffic is not None and time_gnn is not None:
                valid.append(r)
        
        if not valid:
            print(f"{level:<8} {'N/A':<5} {'N/A':<8} {'N/A':<6} {'0':<6}")
            continue
        
        gnn_wins = sum(1 for r in valid if str(r['winner']).upper() == 'GNN')
        traffic_wins = sum(1 for r in valid if str(r['winner']).upper() == 'TRAFFIC')
        ties = sum(1 for r in valid if str(r['winner']).upper() == 'TIE')
        total = len(valid)
        
        # Correct time diff: traffic - gnn (positive = GNN faster)
        diffs = [parse_float(r['time_traffic_s']) - parse_float(r['time_gnn_s']) for r in valid]
        avg_diff = sum(diffs) / len(diffs) if diffs else 0
        
        # Average route duration
        durations = [parse_float(r['avg_route_duration_minutes']) for r in valid 
                    if parse_float(r['avg_route_duration_minutes']) is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Relative improvement (positive = GNN better)
        avg_traffic = sum(parse_float(r['time_traffic_s']) for r in valid) / len(valid) if valid else 0
        avg_gnn = sum(parse_float(r['time_gnn_s']) for r in valid) / len(valid) if valid else 0
        rel_imp = (avg_traffic - avg_gnn) / avg_traffic * 100 if avg_traffic > 0 else 0
        
        print(f"{level:<8} {gnn_wins:<5} {traffic_wins:<8} {ties:<6} {total:<6} "
              f"{avg_diff:>+10.2f}  {avg_duration:>12.2f}  {rel_imp:>+8.2f}")
    
    print("="*130)
    
    # Compute times summary
    print("\n" + "="*90)
    print("COMPUTATION TIMES (ms) - AVERAGES ACROSS ALL DATA")
    print("="*90)
    
    all_valid = []
    for r in rows:
        time_traffic = parse_float(r['time_traffic_s'])
        time_gnn = parse_float(r['time_gnn_s'])
        if time_traffic is not None and time_gnn is not None:
            all_valid.append(r)
    
    for method in ['shortest', 'traffic', 'gnn', 'classifier']:
        key = f'compute_time_{method}_s'
        times_ms = [parse_float(r[key]) * 1000 for r in all_valid 
                   if parse_float(r[key]) is not None]
        
        if times_ms:
            avg = sum(times_ms) / len(times_ms)
            min_t = min(times_ms)
            max_t = max(times_ms)
            print(f"  {method:>12}: avg={avg:>8.2f}ms, min={min_t:>8.2f}ms, max={max_t:>8.2f}ms")
    
    print("="*90)


if __name__ == "__main__":
    main()
