from pathlib import Path

import json
import numpy as np
import pandas as pd
from energypy.sampling import episode
from energypy.memory import Buffer
from collections import defaultdict

import energypylinear
from energypy import checkpoint
import energypy

from energypy.datasets import NEMDataset


def extract_data_from_buffer(buffer, name):
    data = pd.DataFrame(np.squeeze(buff.data[name]))
    data.columns = [f'{name}-{n}' for n in range(data.shape[1])]
    return data


home = Path.cwd() / 'dataset' / 'test-episodes'
fis = [p for p in home.iterdir() if p.suffix == '.csv']

linear = energypylinear.Battery(
    power=2,
    capacity=4,
    efficiency=0.9,
)

#  find latest checkpoint and load
run = './experiments/battery/second'

def sort_fn(path):
    path = str(path).split('-')[-1]
    return int(path)

cp = sorted(checkpoint.get_checkpoint_paths(run), key=sort_fn)[-1]

#  evaluate step
cps = checkpoint.load(run, full=False)

evaluate = []
n_eps = 100
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

cp = checkpoint.load_checkpoint(cp)

actor = cp['nets']['actor']
hyp = cp['hyp']

for fi in fis[:]:
    print(f' {fi.name}')
    path = Path.cwd() / 'results' / fi.stem
    path.mkdir(exist_ok=True, parents=True)

    data = pd.read_csv(fi, index_col=0)
    data.to_csv(path / 'dataset.csv')

    prices = data.loc[:, 'price [$/MWh]'].iloc[:48]
    assert prices.shape[0] == 48

    # def run_lp
    linear_path = path / 'linear.csv'
    if not linear_path.exists():
        print(f' running linear program loading {linear_path}')
        linear_path.parent.mkdir(exist_ok=True, parents=True)
        linear_results = pd.DataFrame(linear.optimize(prices, initial_charge=0, timestep='30min'))
        linear_results.to_csv(linear_path, index=False)
    else:
        print(f' not running linear program loading {linear_path}')
        linear_results = pd.read_csv(linear_path)

    #  def run_rl()
    #  TODO this can be done in parallel
    #  instead of stacking, indexing list
    n_batteries = 1
    dataset = NEMDataset(
        train_episodes=[fi]*n_batteries,
        test_episodes=[fi]*n_batteries,
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

    buff = Buffer(env.elements, size=48)
    counters = defaultdict(int)

    env.setup_test()
    rewards = episode(
        env=env,
        buffer=buff,
        actor=actor,
        hyp=hyp,
        counters=counters,
        rewards=[],
        mode='test'
    )

    #  end of parallelism

    #  for result in results

    rl_results = []
    for name in ['action', 'reward']:
        rl_results.append(extract_data_from_buffer(buff, name))
    rl_results = pd.concat(rl_results, axis=1)

    scale = hyp['reward-scale']
    reward_cols = [c for c in rl_results.columns if 'reward' in c]

    rl_results.loc[:, reward_cols] = rl_results.loc[:, reward_cols] * scale

    rl_results.index = prices.index
    rl_results['price [$/MWh]'] = prices

    rl_results.to_csv(path / 'rl.csv')

    results = {
        'linear-reward': float(linear_results['Actual [$/30min]'].sum()*-1),
        'rl-reward': float(rl_results['reward-0'].sum())
    }

    (path / 'results.json').write_text(json.dumps(results))
