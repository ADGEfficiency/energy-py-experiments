from collections import defaultdict
import json
from pathlib import Path

import pandas as pd

import energypy
from energypy import memory, episode

# dataset_name, debug = cli()
import click

@click.command()
@click.argument('dataset', type=click.Path(exists=True))
@click.argument('hyp', type=click.Path(exists=True))
def cli(dataset, hyp):
    return {
        'hyp': hyp,
        'dataset': dataset
    }

args = cli()
hyp = json.loads(Path(args['hyp']).read_text())
train_eps = [d for d in (args['dataset'] / "train" / "linear").iterdir() if d.suffix == ".json"]

buffer = None
for ep in train_eps:
    print(ep)
    linear_results = json.loads(ep.read_text())
    linear_episode = pd.read_parquet(ep.with_suffix(".parquet"))

    def load_rl_episode(dataset):
        #  dataset = ./attention-dataset/train/2020-01-01
        dims = [p for p in dataset.iterdir() if p.suffix == '.npy']

        #  {prices: np.array, features: np.array, mask: np.array}
        return {p.name: np.load(p) for p in dims}

    # rl_episode = pd.read_parquet(dataset / 'train' / ep.name).with_suffix('.parquet')
    #  here, we need to load features, mask, prices
    rl_episode = load_rl_episode(dataset / 'train' / ep.name)

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
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(linear_results))
    print(linear_results['rl_episode_reward'], linear_results['cost'], linear_results['rl_episode_reward'] + linear_results['cost'])

#  save the buffer
memory.save(buffer, './pretrain/buffer.pkl')
assert buffer.full
