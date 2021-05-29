import argparse
from pathlib import Path

from sklearn.preprocessing import QuantileTransformer, FunctionTransformer, PowerTransformer
import pandas as pd
import numpy as np


def inspect(data):
    stats = data.describe().loc[['mean', 'min', 'max'], :].iloc[:5]
    print(f' shape: {data.shape}')
    print(f' feature statistics: {stats}')

    stats = data.describe().loc[['mean', 'min', 'max'], :].iloc[-5:]
    print(f' shape: {data.shape}')
    print(f' feature statistics: {stats}')


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=0, nargs='?')
    parser.add_argument('--name', default='dataset', nargs='?')
    args = parser.parse_args()
    debug = bool(args.debug)

    subset = None
    horizons = 48
    if debug == True:
        print(' debug mode')
        subset = 2

    return debug, subset, horizons, args.name


def load_nem_data(subset=None):
    home = Path.home() / 'nem-data' / 'data' / 'trading-price'
    fis = [p / 'clean.csv' for p in home.iterdir()]
    fis = [pd.read_csv(p, index_col=0, parse_dates=True) for p in fis]

    if subset:
        fis = fis[-int(subset):]
        print(f'subset to {len(fis)}')

    cols = ['interval-start', 'trading-price', 'REGIONID']
    data = [d[cols] for d in fis]
    data = pd.concat(data, axis=0)

    mask = data['REGIONID'] == 'SA1'
    data = data.loc[mask, :]

    data = data.set_index('interval-start')
    data.index = pd.to_datetime(data.index)
    data = data[~data.index.duplicated(keep='first')]

    print(f' nem dataset shape: {data.shape}')
    return data.sort_index()


def split_train_test(data, split=0.8):
    train_split = int(0.8 * data.shape[0])
    train = data.iloc[:train_split, :]
    test = data.iloc[train_split:, :]
    assert train.shape[0] > test.shape[0]
    return train, test


def create_horizons(data, horizons=48, col='trading-price'):
    print(f' creating {horizons} horizons of {col}')
    features = [data[col].shift(-h) for h in range(horizons)]
    features = pd.concat(features, axis=1)
    features.columns = [f'h-{n}-{col}' for n in range(features.shape[1])]
    return features


def quantile_transform(data, enc, stage='test', debug=False):
    if stage == 'train':
        n_quantiles = data.shape[0]
        if debug:
            n_quantiles = 16

        dist = 'uniform'
        print(f' transforming features train, n_quantiles {n_quantiles}, dist {dist}')
        enc['quantile'] = QuantileTransformer(
            output_distribution=dist,
            n_quantiles=data.shape[0],
            subsample=data.shape[0]
        )
        trans = enc['quantile'].fit_transform(data)
 
    else:
        print(f' transforming features, test {enc["quantile"].n_quantiles_}')
        assert stage == 'test'
        trans = enc['quantile'].transform(data)

    data.loc[:, :] = trans
    inspect(data)
    return data, enc

def log_transform(data, enc, stage='train', debug=False):

    if stage == 'train':
        print(f' transforming log features, train')
        enc['log'] = PowerTransformer()
        trans = enc['log'].fit_transform(data)
    else:
        print(f' transforming log features, test')
        assert stage == 'test'
        trans = enc['log'].transform(data)

    data.loc[:, :] = trans
    inspect(data)
    return data, enc


def make_days(df):
    return pd.date_range(start=df.index[0], end=df.index[-1], freq='d')


def sample_date(date, data):
    start = date
    end = date + pd.Timedelta('24h 00:30:00')
    mask = (data.index >= start) * (data.index < end)

    #  49 because we need that last step for the state, next_state
    if mask.sum() == 49:
        return data.loc[mask, :]

def get_mask_val(data):
    return data.loc[:, 'h-1-trading-price'].min() - 0.5


def mask(data, mask_val):
    #  masking of prices that are in the next day
    prices = data.iloc[:, 0].to_frame()
    mask = create_horizons(
        prices,
        horizons=horizons,
        col=prices.columns[0],
    )
    mask = mask.isnull()
    data.values[mask] = mask_val
    return data


def make_time_features(data):
    time = [t / data.shape[0] for t in range(data.shape[0])]
    #  avoid chained assignment error
    data = data.copy()
    data.loc[:, 'time-to-go'] = time
    return data


def make_price_features(data):
    data['medium-price'] = 0
    mask = data['price [$/MWh]'] > 300
    data.loc[mask, 'medium-price'] = 1

    data['high-price'] = 0
    mask = data['price [$/MWh]'] > 800
    data.loc[mask, 'high-price'] = 1

    return data

if __name__ == '__main__':
    debug, subset, horizons, ds_name = cli()
    data = load_nem_data(subset=subset)
    train, test = split_train_test(data, split=0.8)
    datasets = (('train', train), ('test', test))

    enc = {}

    for name, data in datasets:
        print(f' processing {name}')

        price = data['trading-price']
        data = create_horizons(data, horizons=horizons, col='trading-price')
        #  drop na here because there is no way no fill in these values
        #  that are horizoned at the end of our dataset
        data = data.dropna(axis=0)

        quants, enc = quantile_transform(data, enc, stage=name, debug=debug)

        logs, enc = log_transform(data, enc, stage=name, debug=debug)

        if name == 'train':
            mask_val = get_mask_val(data)

        quants = mask(quants, mask_val)
        logs = mask(logs, mask_val)

        features = pd.concat([quants, logs, price], axis=1)
        features = features.dropna(axis=0).rename({'trading-price': 'price [$/MWh]'}, axis=1)
        assert features.isnull().sum().sum() == 0

        days = sorted(make_days(features))

        print(f' start: {days[0]} end: {days[-1]} num: {len(days)}')
        for day in days[:]:
            ds = sample_date(day, features)

            #  sample_date returns None if data isn't correct length
            if ds is not None:
                assert ds.isnull().sum().sum() == 0

                path = Path.cwd() / ds_name / f'{name}-episodes'
                path.mkdir(exist_ok=True, parents=True)
                ds = make_time_features(ds)
                ds = make_price_features(ds)

                day = day.strftime('%Y-%m-%d')
                ds.to_csv(path / f'{day}.csv')
