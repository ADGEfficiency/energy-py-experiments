all: experiment

req: ./requirements.txt
	pip3 install -q -r requirements.txt

./energy-py/README.md:
	rm -rf ./energy-py
	git clone git@github.com:ADGEfficiency/energy-py
	pip3 install -e energy-py/.
	pip3 install -q -r energy-py/requirements.txt

./nem-data/README.md:
	rm -rf ./nem-data
	git clone git@github.com:ADGEfficiency/nem-data
	pip3 install nem-data/.
	pip3 install -q -r nem-data/requirements.txt

./energy-py-linear/README.md:
	rm -rf ./energy-py-linear
	git clone git@github.com/ADGEfficiency/energy-py-linear
	pip3 install energy-py-linear/.
	pip3 install -q -r energy-py-linear/requirements.txt

#  download aussie electricity price data
~/nem-data/data/TRADINGPRICE/2020-12/clean.parquet: ./nem-data/README.md
	nem -s 2015-01 -e 2020-12 -r trading-price

./datasets/dense/: req
	python3 create_datasets.py --name dense

#  create ML datasets from our price data
./datasets/attention/: req
	python3 create_datasets.py --name attention

experiment: ./energy-py-linear/README.md ./energy-py/README.md ~/nem-data/data/TRADINGPRICE/2020-12/clean.parquet ./datasets/dense/ ./datasets/attention/
