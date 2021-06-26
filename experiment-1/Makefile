setup:
	pip3 install -r requirements.txt
	git clone https://github.com/ADGEfficiency/nem-data
	pip3 install nem-data/.
	nem -s 2014-01 -e 2020-12 -r trading-price

dataset:
	python3 create_dataset.py

linear:
	git clone https://github.com/ADGEfficiency/energy-py-linear
	pip3 install energy-py-linear/.
	python3 linear.py
