from energypy.init import init_fresh
from energypy.main import main

from pathlib import Path


if __name__ == '__main__':
    hyp = {
        "initial-log-alpha": 0.0,
        "gamma": 0.99,
        "rho": 0.995,
        "buffer-size": 1000000,
        "reward-scale": 500,
        "lr": 3e-4,
        "batch-size": 1024,
        "n-episodes": 20000,
        "test-every": 250,
        "n-tests": "all",
        "size-scale": 8,
        "run-name": "rl",
        # "buffer": "experiments/battery/random.pkl",
        "env": {
          "name": "battery",
          "initial_charge": 0.0,
          "episode_length": 48,
          "n_batteries": 16,
          "initial_charge": "random",
          "dataset": {
            "name": "nem-dataset",
            "train_episodes": str(Path.cwd() / 'dataset' / 'train-episodes'),
            "test_episodes": str(Path.cwd() / 'dataset' / 'test-episodes'),
          },
        },
    }

    main(**init_fresh(hyp))
