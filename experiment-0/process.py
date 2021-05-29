import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

import energypy
from energypy import checkpoint
from energypy.agent.memory import Buffer
from energypy.datasets import NEMDataset
from energypy.sampling import episode

from linear import run_linear


def extract_data_from_buffer(buffer, name):
    data = pd.DataFrame(np.squeeze(buff.data[name]))
    assert data.shape[0] % 48 == 0
    data.columns = [f'{name}-{n}' for n in range(data.shape[1])]
    return data

parser = argparse.ArgumentParser()
parser.add_argument('run', default='fifth', nargs='?')
args = parser.parse_args()
run = args.run

home = Path.cwd() / 'dataset' / 'test-episodes'
fis = [p for p in home.iterdir() if p.suffix == '.csv']

#  find latest checkpoint and load
run = f'./experiments/battery/{run}'

#  evaluate step
cps = checkpoint.load(run, full=False)

evaluate = []
for cp in cps:
    res = defaultdict(list)
    for name, rews in cp['rewards'].items():
        res[f'{name}'].append(np.mean(rews))

    res = pd.DataFrame(res)
    res['checkpoint'] = cp['path'].name
    res['path'] = cp['path']
    evaluate.append(res)

evaluate = pd.concat(evaluate, axis=0).sort_values('test-reward', ascending=False)
cp = evaluate.iloc[0].loc['path']
print(evaluate.head())
print(f'loading best checkpoint from {cp}')

#  only want to load two things here...
cp = checkpoint.load_checkpoint(cp)

actor = cp['nets']['actor']
hyp = cp['hyp']

#  parallel stuff first

#  somethings need to be done in sequence

from collections import defaultdict

results = defaultdict(list)

for fi in fis[:]:

    #  run env
    n_batteries = 1
    dataset = NEMDataset(
        train_episodes=[fi, ],
        test_episodes=[fi, ],
        n_batteries=n_batteries
    )
    env = energypy.envs.battery.Battery(
        n_batteries=n_batteries,
        power=2,
        capacity=4,
        efficiency=0.9,
        initial_charge=0,
        dataset=dataset,
        episode_length=48
    )

    buff = Buffer(env.elements, size=hyp['env']['episode_length'])
    counters = defaultdict(int)

    env.setup_test()
    assert not buff.full
    rewards = episode(
        env=env,
        buffer=buff,
        actor=actor,
        hyp=hyp,
        counters=counters,
        rewards=[],
        mode='test'
    )
    assert buff.full

    rl_results = []
    for name in ['action', 'reward', 'observation']:
        rl_results.append(extract_data_from_buffer(buff, name))

    rl_results = pd.concat(rl_results, axis=1)
    episode_length = hyp['env']['episode_length']
    assert rl_results.shape[0] % episode_length == 0

    #  make dir
    print(f' {fi.name}')
    path = Path.cwd() / 'results' / fi.stem
    path.mkdir(exist_ok=True, parents=True)

    #  load dataset, save, get prices
    data = pd.read_csv(fi, index_col=0)
    data.to_csv(path / 'dataset.csv')
    prices = data.loc[:, 'price [$/MWh]'].iloc[:48]

    linear_results = run_linear(fi)

    results['linear-cost'].append(
        float(linear_results['Actual [$/30min]'].sum())
    )

    ep = rl_results
    ep.index = prices.index

    # TODO check
    ep['price [$/MWh]'] = prices

    scale = hyp['reward-scale']
    reward_cols = [c for c in rl_results.columns if 'reward' in c]
    ep.loc[:, reward_cols] = ep.loc[:, reward_cols] * scale

    ep.to_csv(path / 'rl.csv')
    results['rl-cost'].append(
        float(ep['reward-0'].sum() * -1)
    )

    results['date'].append(str(fi).split('/')[-1].split('.')[0])

pd.DataFrame(results).to_csv(path.parent / 'results.csv', index=False)
