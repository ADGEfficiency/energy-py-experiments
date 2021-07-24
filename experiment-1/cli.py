import argparse


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=0, nargs='?')
    parser.add_argument('--name', default='dataset', nargs='?')
    args = parser.parse_args()
    debug = bool(args.debug)
    return args.name, debug
