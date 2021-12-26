.PHONY: install ml-datasets

all: linear

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

#  create ML datasets from our price data - one for a dense net, the other for attention
# ./data/dense/test/features/2020-12-30.parquet: python-setup external-datasets
# 	python3 create_datasets.py dense

./data/attention/test/features/2020-12-30.npy:
	python3 create_datasets2.py attention

ml-datasets: ./data/attention/test/features/2020-12-30.npy

#  run the linear program over our attention net data

linear: ml-datasets
	python3 linear.py attention

./data/linear/
