from pathlib import Path
import json
import numpy as np

import click
import pandas as pd

import energypylinear as epl


def scale_actions(actions, power):
    return (actions / power).tolist()


def main(dataset):

    battery = epl.Battery(power=2, capacity=4, efficiency=0.9)
    for stage in ["train", "test"]:
        raw_path = Path.cwd() / "data" / dataset / stage

        path = Path.cwd() / "data" / "linear" / stage

        for day in [d for d in (raw_path / "prices").iterdir() if d.suffix == ".npy"]:

            prices = np.load(day).reshape(-1)

            res = pd.DataFrame(battery.optimize(prices, initial_charge=0, freq="30T"))
            fi = path / f"{day.stem}.json"
            fi.parent.mkdir(exist_ok=True, parents=True)
            print(f" saving linear results to {fi}")
            fi.write_text(
                json.dumps(
                    {
                        "cost": float(res.sum()["Actual Cost [$/30T]"]),
                        "price": res["Price [$/MWh]"].tolist(),
                        "action": res["Gross [MW]"].tolist(),
                        "scaled-action": scale_actions(res["Gross [MW]"], 2.0),
                    }
                )
            )
            res.to_parquet(fi.with_suffix(".parquet"))


@click.command()
@click.argument("dataset")
def cli(dataset):
    main(dataset)

if __name__ == "__main__":
    cli()
