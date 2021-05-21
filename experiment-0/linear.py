from pathlib import Path
from ray.util.multiprocessing import Pool
import pandas as pd
import energypylinear

def f(index):
        return index

def run_linear(fi):
    linear_path = (Path.cwd() / 'linear' / fi.stem).with_suffix('.csv')
    if not linear_path.exists():
        print(f' running linear program {linear_path}')
        linear_path.parent.mkdir(exist_ok=True, parents=True)

        prices = pd.read_csv(fi, index_col=0).loc[:, 'price [$/MWh]'].iloc[:48]

        linear = energypylinear.Battery(
            power=2,
            capacity=4,
            efficiency=0.9,
        )

        linear_results = pd.DataFrame(
            linear.optimize(prices, initial_charge=0, timestep='30min')
        )
        linear_results.to_csv(linear_path, index=False)
        print(f' linear program run {linear_path}')
        linear_path.parent.mkdir(exist_ok=True, parents=True)
    else:
        print(f' not running linear program loading {linear_path}')
        linear_results = pd.read_csv(linear_path)

    return linear_results


def run_ray(fis):
    pool = Pool(processes=1)
    for result in pool.map(run_linear, fis):
        print(result)


def run(fis):
    for fi in fis:
        run_linear(fi)


if __name__ == '__main__':
    home = Path.cwd() / 'dataset' / 'test-episodes'
    fis = [p for p in home.iterdir() if p.suffix == '.csv']

    run(fis)
