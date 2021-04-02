from pathlib import Path

import pandas as pd


def create_horizons(data, horizons=48):

    features = []
    for horizon in range(horizons):
        features.append(
            data['trading-price'].shift(-horizon)
        )

    features = pd.concat(features, axis=1)
    features.columns = [
        f'{horizon}-{n}' for n in range(features.shape[1])
    ]
    return features


def make_days(df):
    return pd.date_range(
        start=df.index[0],
        end=df.index[-1],
        freq='d'
    )


def sample_date(date, data):
    start = date
    end = date + pd.Timedelta('24h 00:30:00')
    mask = (data.index >= start) * (data.index < end)
    if mask.sum() == 49:
        return data.loc[mask, :]


if __name__ == '__main__':
    home = Path.home() / 'nem-data' / 'trading-price'
    fis = [p / 'clean.csv' for p in home.iterdir()]
    fis = [pd.read_csv(p, index_col=0, parse_dates=True) for p in fis]

    cols = ['interval-start', 'trading-price', 'REGIONID']
    data = [d[cols] for d in fis]
    data = pd.concat(data, axis=0)

    mask = data['REGIONID'] == 'SA1'
    data = data.loc[mask, :]

    data = data.set_index('interval-start')
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    train_split = int(0.8 * data.shape[0])

    train = data.iloc[:train_split, :]
    test = data.iloc[train_split:, :]

    datasets = {'train': train, 'test': test}
    for name, data in datasets.items():
        days = make_days(data)

        for day in days:
            ds = sample_date(day, data)

            if ds is not None:
                path = Path.cwd() / 'dataset' / f'{name}-episodes'
                path.mkdir(exist_ok=True, parents=True)
                day = day.strftime('%Y-%m-%d')

                features = create_horizons(ds)
                features.loc[:, 'price [$/MWh]'] = ds['trading-price']
                features.to_csv(path / f'{day}.csv')

    #  could also save the full csv....


