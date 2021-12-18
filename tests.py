from create_datasets import create_horizons


def test_create_horizions():
    data = pd.DataFrame(
        data=[50, 30, 60, 20, 70, 10],
        columns=["col"],
    )
    horizons = 3
    features = create_horizons(data, horizons, col="col")
    assert features.shape == (6, 3)
    np.testing.assert_array_equal(features.iloc[0, :], [50, 30, 60])
    np.testing.assert_array_equal(features.iloc[-3, :], [20, 70, 10])
