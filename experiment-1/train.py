import json
from pathlib import Path

from energypy.init import init_fresh
from energypy.main import main


if __name__ == '__main__':
    hyp = json.reads((Path.cwd() / 'train.json')).read_text()
    main(**init_fresh(hyp))
