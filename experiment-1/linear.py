import energypylinear as epl
import pandas as pd
from pathlib import Path

battery = epl.Battery(power=2, capacity=4, efficiency=0.9)

home = Path.cwd() / 'linear' / 'train'
home.mkdir(exist_ok=True, parents=True)
#  train
for day in [d for d in (Path.cwd()  / 'dataset' / 'train').iterdir() if d.suffix == '.parquet']:
    data = pd.read_parquet(day)
    prices = data.loc[:, 'price']
    res = pd.DataFrame(battery.optimize(prices, initial_charge=0, timestep='30min'))
    fi = home / f'{day.stem}.json'
    print(f' linear results save to {fi}')

    import json
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
