from pathlib import Path
import json

for ep in [d for d in (Path.cwd()  / 'linear' / 'train').iterdir() if d.suffix == '.json'][:1]:
    linear = json.loads(ep.read_text())
    import pandas as pd
    data = pd.read_parquet(ep.with_suffix('.parquet'))
    hyp = json.loads((Path.cwd() / 'train.json').read_text())

    import energypy
    policy = energypy.make(
        'fixed-policy',
        env=None,
        actions=linear['scaled-action']
    )

    hyp['env']['dataset'] = {
        'name': 'nem-dataset',
        'train_episodes': [ep, ],
        'test_episodes': [ep, ],
    }
    hyp['env']['n_batteries'] = 1

    env = energypy.make(**hyp['env'])

    from energypy import memory
    from collections import defaultdict
    buffer = memory.make(env, {'buffer-size': 1000000})
    results = episode(
        env,
        buffer,
        policy,
        hyp,
        counters=defaultdict(int),
        mode='train'
    )



    #  save the buffer




