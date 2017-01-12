#!/usr/bin/Make -f

export SHELL=/bin/bash
SAMPLES_PER_EPOCH=7000
EPOCHS=5
BATCH_SIZE=1000

.PHONY: environment docs validate simulator telemetry clean cleandocs cleandata cleanmodel cleansimulators

# Phony targets

environment: 
	conda env create -f environment.yml

docs: Makefile.svg

validate: telemetry simulator

simulator: simulator-linux simulator-beta
	"simulator-beta/Dominique Development Linux desktop 64-bit.x86_64"

telemetry: model.h5
	python drive.py model.json

clean: cleandocs cleandata cleanmodel cleansimulators

cleandocs:
	rm -f end-to-end-dl-using-px.pdf
	rm -f Makefile.svg

cleandata:
	rm -rf data
	rm -f data.zip

cleanmodel:
	rm -f model.json
	rm -f model.h5

cleansimulators:
	rm -rf simulator-linux
	rm -rf simulator-beta
	rm -rf simulator-linux.zip
	rm -rf simulator-beta.zip

# File targets

model.h5: model.json

model.json: data/driving_log_train.csv data/driving_log_validation.csv
	python model.py "data/driving_log_train.csv" "data/" $(SAMPLES_PER_EPOCH) $(EPOCHS) $(BATCH_SIZE)

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
	wget "https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf"

drive.py:
	wget -O - "https://d17h27t6h515a5.cloudfront.net/topher/2017/January/586c4a66_drive/drive.py" | dos2unix > $@

Makefile.svg:
	cat Makefile | python makefile2dot.py | dot -Tsvg > $@

simulator-linux.zip:
	wget -O $@ "https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip"

simulator-beta.zip:
	wget -O $@ "https://d17h27t6h515a5.cloudfront.net/topher/2017/January/587527cb_udacity-sdc-udacity-self-driving-car-simulator-dominique-development-linux-desktop-64-bit-5/udacity-sdc-udacity-self-driving-car-simulator-dominique-development-linux-desktop-64-bit-5.zip"

simulator-linux: simulator-linux.zip
	unzip -d $@ -u $< > /dev/null 2>&1

simulator-beta: simulator-beta.zip
	unzip -d $@ -u $< > /dev/null 2>&1
