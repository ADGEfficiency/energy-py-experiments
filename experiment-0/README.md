# Experiment 1 - Perfect Foresight

1. Create dataset of SA trading prices
- episodes are single day, start at 00:00
- requires data downloaded using `nem-data`
- creates test & train episodes in `./dataset`, each episode is a CSV
- basic features of perfect foresight prices (scaled) & time to go

todo
- create horizons on a daily basis
- OR just create the masks properly & apply later?
- create a feature for high prices, negative prices? (current period only)

```
$ pip install -r requirements.txt
$ git clone https://github.com/ADGEfficiency/nem-data
$ nem -s 2014-01 -e 2020-12 -r trading-price
$ py create_dataset.py
```

2. Run dataset on RL
- uses `energypy` - especially `energypy.datasets.NEMDataset`
- output to `./experiments`

```
$ py train.py
```

1. Process results
- runs linear program on each test episode
- runs trained RL battery on each episode
- output to `./results`

TODO
- shouldn't re run linear program if already have results
- maybe run the linear program in a separate py?

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

Fourth `commit f22dcf0d675082d634621363c7e02e4eb71e3262`
- start from 2014
- starting f

Fifth 
- masking to zero of next day prices (done propely now!)
- no improvement

Sixth
- idea is that we arent exploring enough
- lower learning rate for alpha 3e-5
- another idea is that neural nets dont have enough capacity 
- increase size scale from 12 to 16

Seventh
- added features of medium & high prices,
- change quantile to uniform
