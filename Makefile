SHELL=/bin/bash
export SHELL
.PHONY: all clean
.ONESHELL:

all: \
	data/driving_log_overtrain.csv \
	data/driving_log_random_sample.csv \
	data/driving_log_all.csv \
	data/driving_log_train.csv \
	data/driving_log_validation.csv \
	end-to-end-dl-using-px.pdf

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

clean:
	rm -rf data
	rm data.zip

end-to-end-dl-using-px.pdf:
	wget https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf