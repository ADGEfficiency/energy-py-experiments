from pathlib import Path
import json

import click
import pandas as pd

import energypylinear as epl


def scale_actions(actions, power):
    return (actions / power).tolist()


@click.command()
@click.argument("dataset")
@click.option("--debug", default=False)
def cli(dataset, debug):
    battery = epl.Battery(power=2, capacity=4, efficiency=0.9)
    for stage in ["train", "test"]:
        path = Path.cwd() / "data" / dataset / stage

        for day in [d for d in (path / "features").iterdir() if d.suffix == ".parquet"]:
            data = pd.read_parquet(day)
            prices = data.loc[:, "price"]
            res = pd.DataFrame(battery.optimize(prices, initial_charge=0, freq="30T"))
            fi = path / "linear" / f"{day.stem}.json"
            fi.parent.mkdir(exist_ok=True)
            print(f" linear results save to {fi}")
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


if __name__ == "__main__":
    cli()
