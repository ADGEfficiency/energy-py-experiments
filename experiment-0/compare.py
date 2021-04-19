from pathlib import Path
import json

import pandas as pd


home = Path.cwd() / 'results'
days = [p for p in home.iterdir() if p.is_dir()]

ds = []
for p in days:
    p = p / 'results.json'
    data = json.loads(p.read_text())

    data['date'] = p.parent.name
    ds.append(data)

ds = pd.DataFrame(ds).loc[:, ['linear-reward', 'rl-reward']]
sums = ds.sum(axis=0)

print(sums['rl-reward'] / sums['linear-reward'])
