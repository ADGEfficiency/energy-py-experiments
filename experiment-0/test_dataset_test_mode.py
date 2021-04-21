

def make_random_df():
    prices = np.random.uniform(0, 1, 48)
    features = np.random.uniform(0, 1, 48*2).reshape(-1, 2)

    return pd.DataFrame({
        'price [$/MWh]': prices,
        'features': features,
    })


if __name__ == '__main__':




    dataset = NEMDataset(
        train_episodes=fi,
        test_episodes=fi,
        n_batteries=4
    )

    env = energypy.envs.battery.Battery(
        n_batteries=4,
        power=2,
        capacity=4,
        efficiency=0.9,
        initial_charge=0,
        dataset=dataset,
        episode_length=48
    )
