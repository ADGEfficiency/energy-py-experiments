.PHONY: install

all: final

SITE_PACKAGES := $(shell pip show pip | grep '^Location' | cut -f2 -d':')
python-setup: $(SITE_PACKAGES)

$(SITE_PACKAGES): requirements.txt
	pip install -r requirements.txt

./energy-py/README.md:
	rm -rf ./energy-py
	git clone git@github.com:ADGEfficiency/energy-py
	cd energy-py; git checkout dev
	pip3 install -e energy-py/.
	pip3 install -q -r energy-py/requirements.txt

./nem-data/README.md:
	rm -rf ./nem-data
	git clone git@github.com:ADGEfficiency/nem-data
	pip3 install nem-data/.
	pip3 install -q -r nem-data/requirements.txt

./energy-py-linear/README.md:
	rm -rf ./energy-py-linear
	git clone git@github.com:ADGEfficiency/energy-py-linear
	pip3 install energy-py-linear/.
	pip3 install -q -r energy-py-linear/requirements.txt

#  download aussie electricity price data
~/nem-data/data/TRADINGPRICE/2020-12/clean.parquet: ./nem-data/README.md
	nem -s 2015-01 -e 2020-12 -r trading-price

#  third party / input datasets
external-datasets: ./energy-py-linear/README.md ./energy-py/README.md ~/nem-data/data/TRADINGPRICE/2020-12/clean.parquet

#  create ML datasets from our price data
./data/attention/test/features/2020-12-30.npy: ~/nem-data/data/TRADINGPRICE/2020-12/clean.parquet
	python3 create_datasets.py attention

#  run the linear program over our data
./data/linear/test/2020-12-30.json: ./data/attention/test/features/2020-12-30.npy
	python3 linear.py attention

#  fill our buffer with experience that mimics our linear program
./data/pretrain/buffer.pkl: ./data/linear/test/2020-12-30.json
	python3 bootstrap_experience.py ./attention.json

#  pretrain our network
./pretrain/run-one/checkpoints/: ./data/pretrain/buffer.pkl
	python3 pretrain.py ./attention.json

final: ./pretrain/run-one/checkpoints/ ./run_pretrain.py
	python3 run_pretrain.py
