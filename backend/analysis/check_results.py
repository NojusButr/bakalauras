
import json
from pathlib import Path
from collections import Counter


def load_experiment(filepath):
    """Load experiment JSON file."""
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


data = get_latest_experiment()

results = data['results']

print('First 10 results:')
for i in range(min(10, len(results))):
    r = results[i]
    print(f'  {i}: data_level="{r["data_level"]}", scenario="{r["scenario"][:40]}"')

print(f'\nLast 10 results:')
for i in range(max(0, len(results)-10), len(results)):
    r = results[i]
    print(f'  {i}: data_level="{r["data_level"]}", scenario="{r["scenario"][:40]}"')

# Count per level
counts = Counter(r['data_level'] for r in results)
print(f'\nTotal results: {len(results)}')
print(f'\nCount per data level:')
for level in data['config']['data_levels']:
    cnt = counts.get(level, 0)
    print(f'  {level:>6}: {cnt:>4}')

# Count by scenario
scenario_counts = Counter(r['scenario'] for r in results)
print(f'\nCount per scenario:')
for scenario in data['config']['scenarios']:
    cnt = scenario_counts.get(scenario, 0)
    print(f'  {scenario:<60}: {cnt:>4}')
