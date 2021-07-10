.PHONY: pulls3

pushs3:
	make pushs3-dataset
	make pushs3-nem
pushs3-dataset:
	zip -r dataset.zip ./dataset
	aws s3 cp dataset.zip s3://energy-py/public/dataset.zip
pushs3-nem:
	cd ~; zip -r nem.zip ./nem-data/data;
	aws s3 cp ~/nem.zip s3://energy-py/public/nem.zip
