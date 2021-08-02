"""
goal = run the linear actions through my energypy env
"""
import numpy as np
import pandas as pd

import energypylinear as epl
from collections import defaultdict

episode_length = 256
#  note the +1
prices = np.random.uniform(-20, 20, episode_length+1)

battery = epl.Battery(power=2, capacity=4, efficiency=0.9)
linear = pd.DataFrame(battery.optimize(prices[:-1], initial_charge=0, timestep='30min'))

from energypy.agent.memory import Buffer
from energypy.envs.battery import Battery
from energypy.datasets import NEMDataset
from energypy.sampling import episode

ep = pd.DataFrame({
    'price': prices
})

#  run episode in env
n_batteries = 1
dataset = NEMDataset(
    train_episodes=[ep, ],
    test_episodes=[ep, ],
    n_batteries=n_batteries,
    price_col='price'
)
env = Battery(
    n_batteries=n_batteries,
    power=2,
    capacity=4,
    efficiency=0.9,
    initial_charge=0,
    dataset=dataset,
    episode_length=episode_length
)

buffer = Buffer(env.elements, size=episode_length)
counters = defaultdict(int)

from energypy.agent import FixedPolicy

from linear import scale_actions

linear_actions = scale_actions(linear['Gross [MW]'].values, 2.0)
actor = FixedPolicy(env, actions=linear_actions)

env.setup_test()
assert not buffer.full
rewards, info = episode(
    env=env,
    buffer=buffer,
    actor=actor,
    hyp={'reward-scale': 1},
    counters=counters,
    mode='test',
    return_info=True
)
assert buffer.full

from compare import extract_datas_from_buffer

rl_results = extract_datas_from_buffer(buffer, episode_length)
rl_results['losses'] = [float(i['losses_power']) for i in info]
rl_results['gross_power'] = [float(i['gross_power']) for i in info]
rl_results['net_power'] = [float(i['net_power']) for i in info]
rl_results['initial_charge'] = [float(i['initial_charge']) for i in info]
rl_results['final_charge'] = [float(i['final_charge']) for i in info]

print('\n')
res = pd.concat([
    linear['Prices [$/MWh]'],
    linear['Gross [MW]'],
    linear['Import [MW]'],
    linear['Export [MW]'],
    linear['Net [MW]'],
    linear['Losses [MW]'],
    linear['Initial charge [MWh]'],
    linear['Final charge [MWh]'],
    linear['Actual [$/30min]'] * -1,
    rl_results['action-0'] * 2.0,
    rl_results['reward-0'],
    rl_results['gross_power'],
    rl_results['net_power'],
    rl_results['losses'],
    rl_results['initial_charge'],
    rl_results['final_charge'],
    pd.DataFrame({'prices': prices})
], axis=1)

res['loss-error'] = res['Losses [MW]'] - res['losses']

pd.set_option('display.float_format', lambda x: '%.4f' % x)
res = res.dropna(axis=0)
print(res.sum())
print(res.sort_values('loss-error'))

def gross_loss_same_time(row):
    gross = row.loc['Gross [MW]']
    loss = row.loc['Losses [MW]']
    if gross > 0 and loss > 0:
        return True

def import_export(row):
    gross = row.loc['Import [MW]']
    loss = row.loc['Export [MW]']
    if gross > 0 and loss > 0:
        return True

print(res.head())
for func in [gross_loss_same_time, import_export]:
    check = res.apply(func, axis=1)
    print(check.sum())

#  check the efficiencies
res['linear-eff-check'] = res['Losses [MW]'] / res['Export [MW]']
res['ep-eff-check'] = res['losses'] / res['gross_power']
res.loc[res['ep-eff-check'] == 0, 'ep-eff-check'] = np.nan

print(res[['linear-eff-check', 'ep-eff-check']].mean())

#  we still have a difference, now only in the costs

res['reward-error'] = res['Actual [$/30min]']- res['reward-0']
print(
    res['Actual [$/30min]'].sum(),
    res['reward-0'].sum(),
    res['reward-error'].sum()
)
import matplotlib.pyplot as plt
f, axes = plt.subplots(nrows=2)

res = res.iloc[:24, :]
res[['Gross [MW]', 'action-0']].plot(ax=axes[0])
res[['Initial charge [MWh]', 'initial_charge']].plot(ax=axes[1])
f.savefig('validate.png')

#  which energy balance is wrong?
res['linear-bal'] = res['Import [MW]'] - (res['Export [MW]']) - (res['Final charge [MWh]'] - res['Initial charge [MWh]']) * 2
res['linear-bal-1'] = (res['Net [MW]'] - res['Losses [MW]']) - (res['Final charge [MWh]'] - res['Initial charge [MWh]']) * 2
res['ep-bal'] = res['gross_power'] - (res['final_charge'] - res['initial_charge']) * 2
res['linear-reward-check'] = np.abs(res['Prices [$/MWh]']) * np.abs(res['Net [MW]']) / 2 - np.abs(res['Actual [$/30min]'])
print(res.head(10))
