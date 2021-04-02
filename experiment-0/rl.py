from energypy.init import init_fresh
from energypy.main import main

from pathlib import Path


if __name__ == '__main__':
    hyp = {
        "initial-log-alpha": 0.0,
        "gamma": 0.99,
        "rho": 0.995,
        "buffer-size": 100000,
        "reward-scale": 5,
        "lr": 3e-4,
        "batch-size": 1024,
        "n-episodes": 20000,
        "test-every": 20,
        "n-tests": 5,
        "size-scale": 6,
        "env": {
          "name": "battery",
          "initial_charge": 1.0,
          "episode_length": 48,
          "n_batteries": 4,
          "dataset": {
            "name": "nem-dataset",
            "train_episodes": str(Path.cwd() / 'dataset' / 'train-episodes'),
            "test_episodes": str(Path.cwd() / 'dataset' / 'test-episodes'),
          },
        },
    }

    main(**init_fresh(hyp))
