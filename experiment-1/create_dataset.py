import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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


def create_horizons(data, horizons=48, col='trading-price'):
    print(f' creating {horizons} horizons of {col}')
    features = [data[col].shift(-h) for h in range(horizons)]
    features = pd.concat(features, axis=1)
    features.columns = [f'h-{n}-{col}' for n in range(features.shape[1])]
    return features


def test_create_horizions():
    data = pd.DataFrame(data=[50, 30, 60, 20, 70, 10], columns=['col',])
    horizons = 3
    features = create_horizons(data, horizons, col='col')
    assert features.shape == (6, 3)
    np.testing.assert_array_equal(features.iloc[0, :], [50, 30, 60])
    np.testing.assert_array_equal(features.iloc[-3, :], [20, 70, 10])

from sklearn.preprocessing import QuantileTransformer, FunctionTransformer, PowerTransformer


def transform(
    data,
    encoders,
    encoder,
    train=True,
    encoder_params={},
):
    if train:
        print(f' fit transforming {encoder}')
        enc = encoders[encoder](**encoder_params)
        data = enc.fit_transform(data)
        encoders[encoder] = enc
        return data, encoders

    print(f' transforming {encoder}')
    enc = encoders[encoder]
    data = enc.transform(data)
    encoders[encoder] = enc
    return data, encoders



if __name__ == '__main__':
    test_create_horizions()
    debug, subset, horizons, ds_name = cli()
    data = load_nem_data(subset=subset)
    train, test = split_train_test(data, split=0.8)
    datasets = (('train', train), ('test', test))
    encoders = {
        'log': PowerTransformer,
        'quantile': QuantileTransformer
    }

    for name, data in datasets:
        print(f' processing {name}, {data.shape}')

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
            train=name == 'train',
        )
        quantile, encoders = transform(
            hrzns,
            encoders,
            'quantile',
            train=name == 'train',
            encoder_params={
                'n_quantiles': hrzns.shape[0],
                'subsample': hrzns.shape[0],
                'output_distribution': 'uniform'
            }
        )

