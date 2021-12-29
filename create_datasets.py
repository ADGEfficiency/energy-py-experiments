from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    QuantileTransformer,
    FunctionTransformer,
    RobustScaler,
    PowerTransformer,
)


def load_nem_data(subset=None):
    home = Path.home() / "nem-data" / "data" / "TRADINGPRICE"
    fis = [p / "clean.csv" for p in home.iterdir()]
    fis = [pd.read_csv(p, index_col=0, parse_dates=True) for p in fis]

    if subset:
        fis = fis[-int(subset) :]
        print(f"subset to {len(fis)}")

    cols = ["interval-start", "RRP", "REGIONID"]
    data = [d[cols] for d in fis]
    data = pd.concat(data, axis=0).rename({"RRP": "price"}, axis=1)

    mask = data["REGIONID"] == "SA1"
    data = data.loc[mask, :]

    data = data.set_index("interval-start")
    data.index = pd.to_datetime(data.index)
    data = data[~data.index.duplicated(keep="first")]

    print(f" nem dataset shape: {data.shape}")
    return data.sort_index()


def split_train_test(data, split=0.8):
    train_split = int(0.8 * data.shape[0])
    train = data.iloc[:train_split, :]
    test = data.iloc[train_split:, :]
    assert train.shape[0] > test.shape[0]
    return train, test


def create_horizons(data, horizons=48, col="trading-price", rename_cols=True):
    print(f"  creating {horizons} horizons of {col}")
    features = [data[col].shift(-h) for h in range(horizons)]
    features = pd.concat(features, axis=1)
    if rename_cols:
        features.columns = [f"h-{n}-{col}" for n in range(features.shape[1])]

    return features


def transform(
    data,
    encoders,
    encoder,
    train=True,
    encoder_params={},
):
    raw = data.copy()
    if train:
        print(f" fit transforming {encoder}")
        enc = encoders[encoder](**encoder_params)
        data = enc.fit_transform(data)

    else:
        print(f" transforming {encoder}")
        enc = encoders[encoder]
        data = enc.transform(data)

    raw.iloc[:, :] = data
    encoders[encoder] = enc

    raw.columns = [c.replace("price", encoder) + "-feature" for c in raw.columns]
    return raw, encoders


def make_days(df):
    return pd.date_range(
        start=df.index.min().replace(hour=0, minute=0),
        end=df.index.max().replace(hour=0, minute=0),
        freq="d",
    )


def sample_date(date, data):
    #  hardcoding that each ep starts at 0000
    start = date.replace(minute=0)
    end = date + pd.Timedelta("24h 00:30:00")
    mask = (data.index >= start) * (data.index < end)

    #  49 because we need that last step for the state, next_state
    if mask.sum() == 49:
        return data.loc[mask, :]


def create_dataset_attention(horizons, subset=None):
    """
    Create features of shape: (batch, seq_len, features)
    Create prices of shape: (batch, seq_len, 1)
    """
    print("\ncreating attention dataset")

    #  has to be local scope
    encoders = {
        "log": PowerTransformer,
        "quantile": QuantileTransformer,
        "robust": RobustScaler,
    }

    data = load_nem_data(subset=subset).drop("REGIONID", axis=1)
    train, test = split_train_test(data, split=0.8)
    datasets = (("train", train), ("test", test))

    for stage, raw in datasets:
        print(f"\n processing {stage}, {data.shape}")

        #  create features that are the future prices (perfect foresight!)
        hrzns = create_horizons(raw, horizons=horizons, col="price")
        #  there is no way no fill in these values that are horizoned at the end of our dataset,
        #  therefore -> drop rows
        hrzns = hrzns.dropna(axis=0)

        #  add a robust encoding of the prices as our features
        robust, encoders = transform(
            hrzns,
            encoders,
            "robust",
            train=stage == "train",
        )

        #  our features are shorter than our prices -> drop rows again
        features = pd.concat([raw, robust], axis=1).dropna(axis=0)
        assert features.isnull().sum().sum() == 0

        days = sorted(make_days(features))
        print(f"  features: start: {days[0]} end: {days[-1]} num: {len(days)}")

        for day in days:
            if (ds := sample_date(day, features)) is None:
                print(f"  {day} not long enough")
            else:
                print(f"  {day} is long enough")
                prices = ds.loc[:, "price"].to_frame()
                feat = ds.drop("price", axis=1)
                mask = (~create_horizons(prices, horizons, "price").isnull()).astype(int)

                #  need to play games to get mask into shape (batch, features, features)
                #  see https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
                seq_len = feat.shape[1]
                feat = feat.values.reshape(-1, seq_len, 1)
                mask = np.repeat(mask.values, seq_len, axis=1).reshape(
                    -1, seq_len, seq_len
                )
                prices = prices.values.reshape(-1, 1)

                path = Path.cwd() / "data" / "attention" / stage
                path.mkdir(exist_ok=True, parents=True)
                day = day.strftime("%Y-%m-%d")

                for name, d in [
                    ("features", feat),
                    ("mask", mask),
                    ("prices", prices),
                ]:
                    p = path / name / f"{day}.npy"
                    p.parent.mkdir(exist_ok=True)
                    np.save(p, d)
                    print(f" saved to {p}")


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


def create_dataset_dense(horizons, subset=None):
    """
    Uses the results of create_dataset_attention

    Create features of shape: (batch, features)
    Create prices of shape: (batch, 1)
    """
    print("creating dense dataset")
    #  has to be local scope
    encoders = {
        "log": PowerTransformer,
        "quantile": QuantileTransformer,
        "robust": RobustScaler,
    }

    data = load_nem_data(subset=subset).drop("REGIONID", axis=1)
    train, test = split_train_test(data, split=0.8)
    datasets = (("train", train), ("test", test))

    for stage, raw in datasets:
        print(f"\n processing {stage}, {data.shape}")
        prices = raw.loc[:, "price"].to_frame()

        #  create features that are the future prices (perfect foresight!)
        hrzns = create_horizons(raw, horizons=horizons, col="price")
        #  there is no way no fill in these values that are horizoned at the end of our dataset,
        #  therefore -> drop rows
        hrzns = hrzns.dropna(axis=0)

        log, encoders = transform(
            hrzns,
            encoders,
            'log',
            train=stage == 'train',
        )
        quantile, encoders = transform(
            hrzns,
            encoders,
            'quantile',
            train=stage == 'train',
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

        features = pd.concat([quantile, log, prices], axis=1).dropna(axis=0)
        assert features.isnull().sum().sum() == 0

        days = sorted(make_days(features))
        print(f"  features: start: {days[0]} end: {days[-1]} num: {len(days)}")

        next_ep_mask = None
        for day in days:
            if (ds := sample_date(day, features)) is None:
                print(f"  {day} not long enough")
            else:
                print(f"  {day} is long enough")

                ds = make_time_features(ds)
                ds = make_price_features(ds)

                prices = ds.iloc[:, 0].to_frame()
                #  create the masks - that mask out next day
                if next_ep_mask is None:
                    print('next ep mask creation')
                    mask = create_horizons(
                        prices,
                        horizons=horizons,
                        col=prices.columns[0],
                        rename_cols=True
                    )
                    next_ep_mask = mask.isnull()

                #  apply the masks
                for col_include in ['log', 'quantile']:
                    cols = [c for c in ds.columns if col_include in c]
                    subset = ds.loc[:, cols]
                    mask_val = mask_vals[cols[0]]
                    subset.values[next_ep_mask] = mask_val
                    ds.loc[:, cols] = subset

                assert ds.isnull().sum().sum() == 0

                path = Path.cwd() / "data" / "dense" / stage
                path.mkdir(exist_ok=True, parents=True)
                day = day.strftime("%Y-%m-%d")

                ds = ds.values.reshape(prices.shape[0], -1)
                prices = prices.values.reshape(prices.shape[0], 1)

                for name, d in [
                    ("features", ds),
                    ("prices", prices),
                ]:
                    p = path / name / f"{day}.npy"
                    p.parent.mkdir(exist_ok=True)
                    np.save(p, d)
                    print(f" saved to {p}")



@click.command()
@click.argument("dataset")
@click.option("--subset", default=None)
@click.option("--horizons", default=24)
def cli(dataset, subset, horizons):
    datasets = {"dense": create_dataset_dense, "attention": create_dataset_attention}
    datasets[dataset](subset=subset, horizons=horizons)


if __name__ == "__main__":
    cli()
