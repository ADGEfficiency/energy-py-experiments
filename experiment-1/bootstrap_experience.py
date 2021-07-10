from collections import defaultdict
import json
from pathlib import Path

import pandas as pd

import energypy
from energypy import memory, episode


hyp = json.loads((Path.cwd() / "train.json").read_text())
train_eps = [d for d in (Path.cwd() / "linear" / "train").iterdir() if d.suffix == ".json"]
buffer = None

for ep in train_eps:
    print(ep)
    linear_results = json.loads(ep.read_text())
    linear_episode = pd.read_parquet(ep.with_suffix(".parquet"))
    rl_episode = pd.read_parquet((Path.cwd() / 'dataset' / 'train' / ep.name).with_suffix('.parquet'))

    hyp["env"]["dataset"] = {
        "name": "nem-dataset",
        "train_episodes": [rl_episode,],
        "test_episodes": [rl_episode,],
        "price_col": "price"
    }
    hyp["env"]["n_batteries"] = 1

    env = energypy.make(**hyp["env"])

    if buffer is None:
        buffer = memory.make(env, {"buffer-size": len(train_eps) * 48})

    policy = energypy.make("fixed-policy", env=env, actions=linear_results["scaled-action"])

    results = episode(env, buffer, policy, hyp, counters=defaultdict(int), mode="train")

    linear_results['rl_episode_reward'] = float(results)
    out = Path.cwd() / 'pretrain' / ep.name
    print(f' write to {out}')
    out.write_text(json.dumps(linear_results))
    print(linear_results['rl_episode_reward'], linear_results['cost'])

#  save the buffer
memory.save(buffer, './linear/buffer.pkl')
assert buffer.full
