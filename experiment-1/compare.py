from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import energypy
from energypy import checkpoint, json_util
from energypy.agent.memory import Buffer
from energypy.datasets import NEMDataset
from energypy.sampling import episode

from run_pretrain import sort_func, evaulate_checkpoints

def extract_data_from_buffer(buffer, name):
    data = pd.DataFrame(np.squeeze(buff.data[name]))
    assert data.shape[0] % 48 == 0
    data.columns = [f'{name}-{n}' for n in range(data.shape[1])]
    return data

#  load linear rewards + episodes
linear = [p for p in (Path.cwd() / 'linear' / 'test').iterdir()
          if p.suffix == '.json']
linear = [json_util.load(p)['cost'] for p in linear]

episodes = [p for p in (Path.cwd() / 'dataset' / 'test').iterdir()
            if p.suffix == '.parquet']

#  load agent from checkpoint
# cp = evaulate_checkpoints('./pretrain/run-one', sort_func)
cp = evaulate_checkpoints('./experiments/battery/nine', sort_func)
cp = checkpoint.load_checkpoint(cp['path'])
print(cp['path'])
actor = cp['nets']['actor']
hyp = cp['hyp']

#  run episodes
results = defaultdict(list)
for ep, linear_cost in zip(episodes[:50], linear):

    #  run episode in env
    n_batteries = 1
    dataset = NEMDataset(
        train_episodes=[ep, ],
        test_episodes=[ep, ],
        n_batteries=n_batteries,
        price_col='price'
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
        mode='test'
    )
    assert buff.full

    #  extract results
    rl_results = []
    for name in ['action', 'reward', 'observation']:
        rl_results.append(extract_data_from_buffer(buff, name))

    rl_results = pd.concat(rl_results, axis=1)
    episode_length = hyp['env']['episode_length']
    assert rl_results.shape[0] % episode_length == 0

    #  make dir
    print(f' {ep.name}')
    path = Path.cwd() / 'results' / ep.stem
    path.mkdir(exist_ok=True, parents=True)

    results['linear_cost'].append(float(linear_cost))
    results['rl_cost'].append(float(rewards))
    results['date'].append(str(ep).split('/')[-1].split('.')[0])

pd.DataFrame(results).to_csv(path.parent / 'results.csv', index=False)
