from collections import defaultdict
import json
from pathlib import Path
from rich.progress import Progress

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
@click.argument('hyp', type=click.Path(exists=True))
def cli(hyp):

    hyp = json.loads(Path(hyp).read_text())
    train_eps = [d for d in (Path.cwd() / 'data' / "linear" / "train").iterdir() if d.suffix == ".json"]
    print(f"loaded {len(train_eps)} train episodes")

    dataset = hyp['dataset']

    n_times_filled = 0
    check = []

    #  make an env to make a memory
    hyp["env"]["n_batteries"] = 1
    env = energypy.make(**hyp["env"])
    buffer = memory.make(env, {
        "buffer-size": hyp['buffer-size']
    })


    #  48 episode len ????
    total = buffer.size / 48

    path = Path.cwd() / 'data' / dataset / 'pretrain'


    with Progress() as progress:
        task = progress.add_task("Filling buffer...", total=total)

        while not buffer.full:
            progress.update(task)
            for ep in train_eps:

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
                    out = path / ep.name
                    out.parent.mkdir(exist_ok=True, parents=True)
                    out.write_text(json.dumps(linear_results))
                    check.append({
                        "rl-reward": linear_results['rl_episode_reward'],
                        "linear-cost": linear_results['cost'],
                        "difference": linear_results['rl_episode_reward'] + linear_results['cost']
                    })

            n_times_filled += 1

    #  save the buffer
    assert buffer.full
    memory.save(buffer, hyp['buffer'])

    #  save the check
    dataset = hyp['dataset']
    pd.DataFrame(check).to_csv('./data/{dataset}/pretrain/check.csv', index=False)


if __name__ == '__main__':
    args = cli()
