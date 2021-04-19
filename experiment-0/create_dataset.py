from pathlib import Path

from sklearn.preprocessing import QuantileTransformer
import pandas as pd
import numpy as np


def load_nem_data():
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
    data = data[~data.index.duplicated(keep='first')]
    return data.sort_index()


def split_train_test(data, split=0.8):
    train_split = int(0.8 * data.shape[0])
    train = data.iloc[:train_split, :]
    test = data.iloc[train_split:, :]
    assert train.shape[0] > test.shape[0]
    return train, test


def create_horizons(data, horizons=48, scale=False):
    features = []
    for horizon in range(horizons):
        features.append(
            data['trading-price'].shift(-horizon)
        )

    features = pd.concat(features, axis=1).dropna(axis=0)
    features.columns = [f'horizon-{n}' for n in range(features.shape[1])]
    return features


def transform_features(data, enc=None, stage='test'):

    if stage == 'train':
        enc = QuantileTransformer(output_distribution='normal')
        trans = enc.fit_transform(data)

    else:
        assert stage == 'test'
        trans = enc.transform(data)

    data.loc[:, :] = trans
    stats = data.describe().loc[['mean', 'min', 'max'], :]
    print(f'feature statistics: {stats}')
    return data, enc


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

    #  49 because we need that last step for the state, next_state
    if mask.sum() == 49:
        return data.loc[mask, :]


def make_time_features(data):
    time = [t / data.shape[0] for t in range(data.shape[0])]
    #  avoid chained assignment error
    data = data.copy()
    data.loc[:, 'time-to-go'] = time
    return data


if __name__ == '__main__':
    data = load_nem_data()

    train, test = split_train_test(data, split=0.8)
    enc = None

    datasets = {'train': train, 'test': test}

    for name, data in datasets.items():
        print(f'processing {name}')

        price = data['trading-price']
        data = create_horizons(data, horizons=4*2)
        data, enc = transform_features(data, enc, stage=name)
        data.loc[:, 'price [$/MWh]'] = price

        assert data.isnull().sum().sum() == 0
        days = make_days(data)

        for day in days[:1]:
            ds = sample_date(day, data)
            #  sample_date returns None if data isn't correct length
            if ds is not None:

                path = Path.cwd() / 'dataset' / f'{name}-episodes'
                path.mkdir(exist_ok=True, parents=True)

                ds = make_time_features(ds)

                day = day.strftime('%Y-%m-%d')
                ds.to_csv(path / f'{day}.csv')
