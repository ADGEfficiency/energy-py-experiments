from energypy import checkpoint


if __name__ == '__main__':
    cps = checkpoint.load(f'./pretrain/run-one')

    #  TODO
    cp = cps[0]

    def sort_func(cp):
        return cp['path'].split('/')[-1].split)

    cps = sorted(cps, )

    from energypy.checkpoint import init_checkpoint

    expt = init_checkpoint(cp['path'])

    from energypy.main import main

    main(**expt)

