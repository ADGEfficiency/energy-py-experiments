import click

from energypy import checkpoint
from energypy.checkpoint import init_checkpoint
from energypy.main import main


def sort_func(cp):
    return int(str(cp['path']).split('/')[-1].split('-')[-1])


def evaulate_checkpoints(run_path, sort_func):
    cps = checkpoint.load(run_path, full=False)
    cps = sorted(cps, key=sort_func)
    return cps[-1]


@click.command()
@click.argument('dataset')
def cli(dataset):
    cp = evaulate_checkpoints(f'./data/{dataset}/pretrain/', sort_func)
    print(f" loaded {cp['path']}")
    expt = init_checkpoint(cp['path'])
    main(**expt)


if __name__ == '__main__':
    cli()
