from create_dataset import *


def create_dataset_attention():
    pass


if __name__ == '__main__':
    horizons = 4

    data = load_nem_data(subset=20).drop('REGIONID', axis=1)
    train, test = split_train_test(data, split=0.8)
    datasets = (('train', train), ('test', test))

    encoders = {
        'log': PowerTransformer,
        'quantile': QuantileTransformer,
        'robust': RobustScaler
    }

    for data_name, data in datasets:
        print(f' processing {data_name}, {data.shape}')

        hrzns = create_horizons(data, horizons=horizons, col='price')
        hrzns = hrzns.dropna(axis=0)

        #  now add transforms
        robust, encoders = transform(
            hrzns,
            encoders,
            'robust',
            train=data_name == 'train',
        )
        features = pd.concat([data, robust], axis=1)

        days = sorted(make_days(features))
        print(f' start: {days[0]} end: {days[-1]} num: {len(days)}')

        mask = create_horizons(features, horizons, 'price').isnull()
        for day in days:
            if (ds := sample_date(day, features)) is not None:
                path = Path.cwd() / 'attention-dataset' / data_name
                path.mkdir(exist_ok=True, parents=True)
                day = day.strftime('%Y-%m-%d')
                p = path / 'features' / f'{day}.parquet'
                p.parent.mkdir(exist_ok=True)
                ds.to_parquet(p)
                print(f' saved to {p}')

                p = path / 'mask' /  f'{day}.parquet'
                p.parent.mkdir(exist_ok=True)
                ds.to_parquet(p)
                print(f' saved to {p}')

