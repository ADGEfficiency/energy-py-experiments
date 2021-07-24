import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer, PowerTransformer


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=0, nargs='?')
    parser.add_argument('--name', default='dataset', nargs='?')
    args = parser.parse_args()
    debug = bool(args.debug)

    subset = None
    horizons = 24
    if debug == True:
        print(' debug mode')
        subset = 32

    return debug, subset, horizons, args.name


def load_nem_data(subset=None):
    home = Path.home() / 'nem-data' / 'trading-price'
    fis = [p / 'clean.csv' for p in home.iterdir()]
    fis = [pd.read_csv(p, index_col=0, parse_dates=True) for p in fis]

    if subset:
        fis = fis[-int(subset):]
        print(f'subset to {len(fis)}')

    cols = ['interval-start', 'trading-price', 'REGIONID']
    data = [d[cols] for d in fis]
    data = pd.concat(data, axis=0).rename({'trading-price': 'price'}, axis=1)

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


def create_horizons(data, horizons=48, col='trading-price', rename_cols=True):
    print(f' creating {horizons} horizons of {col}')
    features = [data[col].shift(-h) for h in range(horizons)]
    features = pd.concat(features, axis=1)
    if rename_cols:
        features.columns = [f'h-{n}-{col}' for n in range(features.shape[1])]
    return features


def test_create_horizions():
    data = pd.DataFrame(data=[50, 30, 60, 20, 70, 10], columns=['col',])
    horizons = 3
    features = create_horizons(data, horizons, col='col')
    assert features.shape == (6, 3)
    np.testing.assert_array_equal(features.iloc[0, :], [50, 30, 60])
    np.testing.assert_array_equal(features.iloc[-3, :], [20, 70, 10])



def transform(
    data,
    encoders,
    encoder,
    train=True,
    encoder_params={},
):
    raw = data.copy()
    if train:
        print(f' fit transforming {encoder}')
        enc = encoders[encoder](**encoder_params)
        data = enc.fit_transform(data)

    else:
        print(f' transforming {encoder}')
        enc = encoders[encoder]
        data = enc.transform(data)

    raw.iloc[:, :] = data
    encoders[encoder] = enc

    raw.columns = [c.replace('price', encoder)+'-feature' for c in raw.columns]
    return raw, encoders


def make_days(df):
    return pd.date_range(start=df.index[0].replace(hour=0, minute=0), end=df.index[-1].replace(hour=0, minute=0), freq='d')


def sample_date(date, data):
    start = date.replace(minute=0)
    end = date + pd.Timedelta('24h 00:30:00')
    mask = (data.index >= start) * (data.index < end)

    #  49 because we need that last step for the state, next_state
    if mask.sum() == 49:
        return data.loc[mask, :]


def make_time_features(data):
    """done on one episode at a time"""
    time = [t / data.shape[0] for t in range(data.shape[0])]
    #  avoid chained assignment error
    data = data.copy()
    data.loc[:, 'time-to-go'] = time
    return data


def make_price_features(data):
    """done on one episode at a time"""
    data['medium-price'] = 0
    mask = data['price'] > 300
    data.loc[mask, 'medium-price'] = 1

    data['high-price'] = 0
    mask = data['price'] > 800
    data.loc[mask, 'high-price'] = 1
    return data


def create_dataset_dense():
    data = load_nem_data(subset=subset)
    train, test = split_train_test(data, split=0.8)
    datasets = (('train', train), ('test', test))
    encoders = {
        'log': PowerTransformer,
        'quantile': QuantileTransformer
    }

    for data_name, data in datasets:
        print(f' processing {data_name}, {data.shape}')

        price = data['price'].to_frame()
        hrzns = create_horizons(data, horizons=horizons, col='price')
        #  there is no way no fill in these values
        #  that are horizoned at the end of our dataset
        #  therefore -> drop rows
        hrzns = hrzns.dropna(axis=0)

        log, encoders = transform(
            hrzns,
            encoders,
            'log',
            train=data_name == 'train',
        )
        quantile, encoders = transform(
            hrzns,
            encoders,
            'quantile',
            train=data_name == 'train',
            encoder_params={
                'n_quantiles': hrzns.shape[0],
                'subsample': hrzns.shape[0],
                'output_distribution': 'uniform'
            }
        )

        mask_vals = {
            'h-0-quantile-feature': quantile.min().min(),
            'h-0-log-feature': log.min().min(),
        }

        assert hrzns.shape[0] == log.shape[0] == quantile.shape[0]

        for d in [hrzns, log, quantile]:
            assert d.isnull().sum().sum() == 0

        features = pd.concat([quantile, log, price], axis=1)

        next_ep_mask = None
        days = sorted(make_days(features))

        print(f' start: {days[0]} end: {days[-1]} num: {len(days)}')
        if debug:
            end = 3
        else:
            end = -1

        for day in days[:end]:

            ds = sample_date(day, features)

            #  sample_date returns None if data isn't correct length
            if ds is None or ds.isnull().sum().sum() != 0:
                pass
            else:
                raw = ds.copy()

                if next_ep_mask is None:
                    print('next ep mask creation')
                    prices = ds.iloc[:, 0].to_frame()
                    mask = create_horizons(
                        prices,
                        horizons=horizons,
                        col=prices.columns[0],
                        rename_cols=True
                    )
                    next_ep_mask = mask.isnull()

                path = Path.cwd() / ds_name / data_name
                path.mkdir(exist_ok=True, parents=True)

                ds = make_time_features(ds)
                ds = make_price_features(ds)

                for col_include in ['log', 'quantile']:
                    cols = [c for c in ds.columns if col_include in c]
                    subset = ds.loc[:, cols]
                    mask_val = mask_vals[cols[0]]
                    subset.values[next_ep_mask] = mask_val
                    ds.loc[:, cols] = subset

                if ds.isnull().sum().sum() != 0:
                    assert 1 == 0
                day = day.strftime('%Y-%m-%d')
                p = path / f'{day}.parquet'
                print(f' saving to {p}')
                ds.to_parquet(p)


if __name__ == '__main__':
    debug, subset, horizons, ds_name, func = cli()

    funcs = {
        'dense': create_dataset_dense,
        'attention': create_dataset_attention,
    }

    func = funcs[func]


