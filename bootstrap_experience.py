from collections import defaultdict
import json
from pathlib import Path

import click
import pandas as pd

import energypy
from energypy import memory, episode
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_rl_episode(day, dataset):
    """load features, mask, prices"""
    # #  {prices: np.array, features: np.array, mask: np.array}
    ep = {}
    for el in ['prices', 'features', 'mask']:
        try:
            ep[el] = np.load(
                Path.cwd() / 'data' / dataset / 'train' / el / f"{day}.npy"
            )
        except FileNotFoundError:
            assert el == 'mask'
            ep[el] = np.ones_like(ep['features'])

            pass
    return ep


@click.command()
@click.argument('dataset')
@click.argument('hyp', type=click.Path(exists=True))
def cli(dataset, hyp):

    hyp = json.loads(Path(hyp).read_text())
    train_eps = [d for d in (Path.cwd() / 'data' / "linear" / "train").iterdir() if d.suffix == ".json"]
    print(f"loaded {len(train_eps)} train episodes")

    n_times_filled = 0
    check = []

    #  make an env to make a memory
    hyp["env"]["n_batteries"] = 1
    env = energypy.make(**hyp["env"])
    buffer = memory.make(env, {
        "buffer-size": hyp['buffer-size']
    })

    while not buffer.full:
        for ep in train_eps:
            print(f"\n{ep}, {100 * buffer.cursor / buffer.size}, {buffer.full}, {n_times_filled}")

            linear_results = json.loads(ep.read_text())
            linear_episode = pd.read_parquet(ep.with_suffix(".parquet"))

            rl_episode = load_rl_episode(ep.stem, dataset)

            hyp["env"]["dataset"] = {
                "name": f"nem-dataset-{dataset}",
                "train_episodes": [rl_episode,],
                "test_episodes": [rl_episode,],
                "price_col": "price"
            }

            #  make a fixed policy that follows the linear program
            policy = energypy.make("fixed-policy", env=env, actions=linear_results["scaled-action"])

            #  run the episode following this policy
            #  this fills the memory
            results = episode(env, buffer, policy, hyp, counters=defaultdict(int), mode="train")

            #  only save the episode results once
            if n_times_filled == 0:
                linear_results['rl_episode_reward'] = float(results)
                out = Path.cwd() / 'data' / 'pretrain' / ep.name
                print(f' saving to {out}')
                out.parent.mkdir(exist_ok=True, parents=True)
                out.write_text(json.dumps(linear_results))
                check.append({
                    "rl-reward": linear_results['rl_episode_reward'],
                    "linear-cost": linear_results['cost'],
                    "difference": linear_results['rl_episode_reward'] + linear_results['cost']
                })

        n_times_filled += 1

    #  save the buffer
    memory.save(buffer, './data/pretrain/initial-buffer.pkl')
    assert buffer.full

    #  save the check
    pd.DataFrame(check).to_csv('./data/pretrain/check.csv', index=False)



if __name__ == '__main__':
    args = cli()
