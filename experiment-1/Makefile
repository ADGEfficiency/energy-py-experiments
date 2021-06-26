setup:
	pip3 install -r requirements.txt
	git clone https://github.com/ADGEfficiency/nem-data
	pip3 install nem-data/.
	nem -s 2014-01 -e 2020-12 -r trading-price
	python3 create_dataset.py
