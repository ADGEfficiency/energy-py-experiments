from pathlib import Path
import json

import pandas as pd

def compare():
    home = Path.cwd() / 'results'
    days = [p for p in home.iterdir() if p.is_dir()]

    ds = []
    for p in days:
        rl = json.loads((p / 'rl.json').read_text())
        linear = json.loads((p / 'linear.json').read_text())
        data['date'] = p.parent.name
        ds.append(data)

    ds = pd.DataFrame(ds).loc[:, ['linear-reward', 'rl-reward', 'date']]
    ds['pct'] = ds['rl-reward'] / ds['linear-reward']
    ds['diff'] = abs(ds['linear-reward'] - ds['rl-reward'])
    sums = ds.sum(axis=0)

    print(sums['rl-reward'] / sums['linear-reward'])
    return ds

compare()
