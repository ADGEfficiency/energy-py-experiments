from collections import defaultdict
from pathlib import Path

import click

from energypy import memory, train, json_util, init, utils, make
from energypy import checkpoint

@click.command()
@click.argument('hyp', type=click.Path(exists=True))
def cli(hyp):
    hyp = json_util.load(hyp)

    env = make(**hyp['env'])
    buffer = memory.load(
        hyp['buffer'],
        # './data/pretrain/initial-buffer.pkl'
    )

    #  very similar to init_fresh
    nets = init.init_nets(env, hyp)
    optimizers = init.init_optimizers(hyp)
    counters = defaultdict(int)
    writer = utils.Writer(
        'pretrain',
        counters,
        './data/pretrain/'
    )

    target_entropy = nets.pop('target_entropy')
    hyp['target-entropy'] = target_entropy

    print(f' buffer len {len(buffer)}')

    #  train on buffer
    epoch_len = int(len(buffer) / hyp['batch-size'])
    global_step = 0
    for epoch in range(10):
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

    #  only save the last step
    checkpoint.save(
        hyp,
        nets,
        optimizers,
        buffer,
        step,
        rewards={"test-reward": 0},
        counters={"count": 0},
        paths=None,
        path=f'./data/pretrain/checkpoints/epoch-{epoch}-step-{step}-cglobal-{global_step}'
    )


if __name__ == '__main__':
    cli()
