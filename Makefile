SHELL=/bin/bash
export SHELL
.PHONY: all clean environment train validate cleanmodel docs
.ONESHELL:
SAMPLES=10
EPOCHS=5

# Phony targets


all: train

environment: 
	conda env create -f environment.yml

docs: end-to-end-dl-using-px.pdf

train: data/driving_log_train.csv
	python model.py "data/driving_log_train.csv" "data/" $(SAMPLES) $(EPOCHS)

validate: model.h5 model.json
	python drive.py

clean: cleanmodel
	rm -rf data
	rm -f data.zip
	rm -f drive.py
	rm -f end-to-end-dl-using-px.pdf

cleanmodel:
	rm -f model.json
	rm -f model.h5

# File targets

model.h5:
	train

model.json:
	train

data/driving_log.csv: data.zip
	unzip -u $< > /dev/null 2>&1
	rm -rf __MACOSX

data.zip:
	wget -nc "https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip"

data/driving_log_all.csv: data/driving_log.csv
	cat $< | tail -n+2 | shuf > $@

data/driving_log_overtrain.csv: data/driving_log_all.csv
	cat <(cat $< | sort -k4 -n -t, | head -n1) <(cat $< | sort -k4 -nr -t, | head -n1) <(cat $< | awk -F, -vOFS=, '{print $$1, $$2, $$3, sqrt($$4*$$4), $$5, $$6, $$7}' | sort -k4 -n -t, | head -n1) > $@

data/driving_log_random_sample.csv: data/driving_log_all.csv
	cat $< | shuf | head > $@

data/driving_log_train.csv: data/driving_log_all.csv
	cat $< | head -n7000 > $@

data/driving_log_validation.csv: data/driving_log_all.csv
	cat $< | tail -n+7000 > $@

end-to-end-dl-using-px.pdf:
	wget https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

drive.py:
	wget -O - https://d17h27t6h515a5.cloudfront.net/topher/2017/January/586c4a66_drive/drive.py | dos2unix > $@
