from collections import defaultdict
from energypy import memory, train, json_util, init, utils, make
from energypy import checkpoint
from pathlib import Path
hyp = json_util.load('./train.json')

hyp["env"]["dataset"] = {
    "name": "nem-dataset",
    "train_episodes": [p for p in Path('./dataset/train').iterdir()],
    "test_episodes": [p for p in Path('./dataset/test').iterdir()],
    "price_col": "price"
}
hyp["env"]["n_batteries"] = 1

env = make(**hyp['env'])
buffer = memory.load('./pretrain/buffer.pkl')

#  very similar to init_fresh
nets = init.init_nets(env, hyp)
optimizers = init.init_optimizers(hyp)
counters = defaultdict(int)
writer = utils.Writer('pretrain', counters, './pretrain/run-one')

target_entropy = nets.pop('target_entropy')
hyp['target-entropy'] = target_entropy

print(f' buffer len {len(buffer)}')

#  train on buffer
epoch_len = int(len(buffer) / hyp['batch-size'])
global_step = 0
for epoch in range(50):
    for step in range(epoch_len):
        print(f' epoch: {epoch}, step: {step} of {epoch_len}')
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
        global_step += 1

    checkpoint.save(
        hyp,
        nets,
        optimizers,
        buffer,
        step,
        rewards={"test-reward": 0},
        counters={"count": 0},
        paths=None,
        path=f'./pretrain/run-one/checkpoints/epoch-{epoch}-step-{step}-cglobal-{global_step}'
    )
