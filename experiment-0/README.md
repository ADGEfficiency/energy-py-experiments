## Experiment 1 - Perfect Foresight

1. Create dataset of SA trading prices
- episodes are single day, start at 00:00
- requires data downloaded using `nem-data`
- creates test & train episodes in `./dataset`, each episode is a CSV
- basic features of perfect foresight prices (scaled) & time to go

```
$ nem -s 2016-01 -e 2020-12 -r trading-price
$ py create_dataset.py
```

2. Run dataset on RL
- uses `energypy` - especially `energypy.datasets.NEMDataset`
- output to `./experiments`

```
$ py train.py
```

3. Process results
- runs linear program on each test episode
- runs trained RL battery on each episode
- output to `./results`

Questions / TODO
- which battery to use (or average across all?)
- is each episode in the test rollouts the same?
- save interval data for comparisons
- want to save last 100 episodes in checkpoint (to select best checkpoint from)

```
$ py process.py
```

4. Compare results

TODO
- use streamlit

What about running more than a day??


## Run results

First
- 58%
- horizon 12 hours
- reward scale 500
- size scale 8

Second
- ? %
- horizon 48 hours, n_quantiles='n_samples'
- reward scale 500
- size scale 12

Third
- use random memory of second
- same params as second
- fix seed at 42
