.PHONY: ~/nem-data/trading-price/

experiment: final

req: ./requirements.txt
	pip3 install -q -r requirements.txt

./nem-data/README.md: ./requirements.txt
	rm -rf ./nem-data
	git clone git@github.com:ADGEfficiency/nem-data
	pip3 install nem-data/.
	pip3 install -q -r nem-data/requirements.txt

./energy-py/:
	rm -rf ./energy-py
	git clone git@github.com:ADGEfficiency/energy-py
	pip3 install -e energy-py/.
	pip3 install -q -r energy-py/requirements.txt

~/nem-data/trading-price/: ./nem-data/README.md
	nem -s 2015-01 -e 2020-12 -r trading-price

# --- DENSE NETWORK WON'T WORK ANYMORE

./attention-dataset/: ./create_dataset_attention.py
	python3 create_dataset.py attention-dataset

./linear/train/: ./dataset/ ./linear.py
	rm -rf ./energy-py-linear
	git clone https://github.com/ADGEfficiency/energy-py-linear
	pip3 install energy-py-linear/.
	python3 linear.py attention-dataset

./pretrain/buffer.pkl: ./energy-py/ ./linear/train/ ./bootstrap_experience.py
	python3 bootstrap_experience.py attention-dataset attention.json

# ./pretrain/run-one/checkpoints/: ./pretrain/buffer.pkl ./pretrain.py
# 	python3 pretrain.py

# final: ./pretrain/run-one/checkpoints/ ./run_pretrain.py
# 	python3 run_pretrain.py
