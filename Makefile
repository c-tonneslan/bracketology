.PHONY: all fetch dataset train charts test clean sim

all: fetch dataset train charts

fetch:
	python3 scripts/fetch.py

dataset:
	python3 scripts/build_dataset.py

train:
	python3 scripts/train.py

charts:
	python3 scripts/charts.py

sim:
	python3 scripts/simulate.py --teams sample_east.txt --n 5000

test:
	python3 -m pytest tests/ -v

clean:
	rm -f data/*.parquet models/* charts/*.png
