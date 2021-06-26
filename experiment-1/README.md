# Experiment 1 - Pretraining on perfect rollouts

## Methods

### 1. Create a dataset of SA trading prices

```python
$ make dataset
```

- one file per day, starting at `00:00`,
- `./dataset/{train,test}/{date}.parquet`

Required columns (that are not used as features):
- `price` - an electricity price in $/MWh

All other columns are treated as the features

Features:
- masked perfect foresight,
- time to go,
- rank one hot enc - https://scottclowe.com/2016-03-05-rank-hot-encoder/
