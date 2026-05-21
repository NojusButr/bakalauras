
"""Breakdown of experiment routes and evaluations."""

import json
from pathlib import Path


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


exp = get_latest_experiment()

config = exp['config']
results = exp['results']

print('='*80)
print('EXPERIMENT CONFIGURATION')
print('='*80)
print(f'\nScenarios ({len(config["scenarios"])}):\n')
for i, s in enumerate(config['scenarios'], 1):
    print(f'  {i}. {s}')

print(f'\nRoutes per scenario: {config["n_route_pairs"]}')
print(f'Data degradation levels: {len(config["data_levels"])}')
print(f'  {config["data_levels"]}')

print(f'\n' + '='*80)
print('EXPECTED vs ACTUAL COUNTS')
print('='*80)

scenarios = config['scenarios']
levels = config['data_levels']
expected_total = len(scenarios) * config['n_route_pairs']

print(f'\nExpected calculation:')
print(f'  {len(scenarios)} scenarios × {config["n_route_pairs"]} routes/scenario')
print(f'  = {expected_total} total routes (base set)')
print(f'\n  {expected_total} routes × {len(levels)} data levels')
print(f'  = {expected_total * len(levels)} total evaluations')

print(f'\nActual results:')
print(f'  Total result entries: {len(results)}')

# Count by scenario and data level
by_scenario = {}
by_level = {}
for r in results:
    scenario = r['scenario']
    level = r['data_level']
    by_scenario[scenario] = by_scenario.get(scenario, 0) + 1
    by_level[level] = by_level.get(level, 0) + 1

print(f'\nBREAKDOWN BY SCENARIO:')
print('-'*80)
total_by_scenario = 0
for scenario in scenarios:
    count = by_scenario.get(scenario, 0)
    per_level = count / len(levels) if count > 0 else 0
    print(f'  {scenario:<60} {count:>4} entries ({per_level:>6.1f} routes per level)')
    total_by_scenario += count

print(f'  {"TOTAL":<60} {total_by_scenario:>4}')

print(f'\nBREAKDOWN BY DATA LEVEL:')
print('-'*80)
for level in levels:
    count = by_level.get(level, 0)
    routes = count / len(scenarios) if count > 0 else 0
    print(f'  Data level {level:>6}: {count:>4} entries ({routes:>6.1f} routes per scenario)')

