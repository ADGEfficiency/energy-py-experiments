from collections import defaultdict
from energypy import memory, train, json_util, init, utils, make


hyp = json_util.load('./train.json')

hyp["env"]["dataset"] = {
    "name": "nem-dataset",
    "train_episodes": ['./dataset/train/2014-01-01.parquet',],
    "test_episodes": ['./dataset/train/2014-01-01.parquet',],
    "price_col": "price"
}
hyp["env"]["n_batteries"] = 1

env = make(**hyp['env'])
buffer = memory.load('./linear/buffer.pkl')

#  very similar to init_fresh

nets = init.init_nets(env, hyp)
optimizers = init.init_optimizers(hyp)
counters = defaultdict(int)
writer = utils.Writer('pretrain', counters, './pretrain/run-one')

target_entropy = nets.pop('target_entropy')
hyp['target-entropy'] = target_entropy

#  train on buffer
epoch_len = len(buffer)
for epoch in range(3):
    for step in range(epoch_len):
        print(f' epoch: {epoch}, step: {step}')
        #  randomly sample each time
        #  not quite right but it's ok enough
        train.train(
            buffer.sample(hyp['batch-size']),
            nets['actor'],
            [nets['online-1'], nets['online-2']],
            [nets['target-1'], nets['target-2']],
            nets['alpha'],
            writer,
            optimizers,
            counters,
            hyp
        )

# episode = step
# checkpoint.save(
#     hyp,
#     nets,
#     optimizers,
#     buffer,
#     episode,
#     rewards,
#     counters,
#     paths
# )
