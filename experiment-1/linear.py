import energypylinear as epl
import pandas as pd
from pathlib import Path
import json

battery = epl.Battery(power=2, capacity=4, efficiency=0.9)
from cli import cli

dataset_name, debug = cli()

for stage in ['train', 'test']:
    for day in [d for d in (Path.cwd()  / dataset_name / stage / 'features').iterdir() if d.suffix == '.parquet']:
        data = pd.read_parquet(day)
        prices = data.loc[:, 'price']
        res = pd.DataFrame(battery.optimize(prices, initial_charge=0, timestep='30min'))
        fi = Path.cwd() / dataset_name / stage / 'linear' / f'{day.stem}.json'
        fi.parent.mkdir(exist_ok=True)
        print(f' linear results save to {fi}')
        fi.write_text(
            json.dumps(
                {
                    "cost": float(res.sum()['Actual [$/30min]']),
                    "price": res['Prices [$/MWh]'].tolist(),
                    "action": res['Gross [MW]'].tolist(),
                    "scaled-action": (res['Gross [MW]'] / 2.0).tolist()
                }
            )
        )
        res.to_parquet(fi.with_suffix('.parquet'))
