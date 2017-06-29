.PHONY: all download


#### common variables ####
HOSTNAME := $(shell hostname)

#### common functions ####
define mkdir_config
	@ echo checking dirctory: $1
	@ [ -d $1 ] || mkdir -p $1
endef

#### bvlc_caffent ####
all:
	@ echo "Do nothing"

download:
	$(call mkdir_config, models/bvlc_reference_caffnet)
	@ python ../tools/download_model_binary.py models/bvlc_reference_caffenet
	@ ./ilsvrc12/get_ilsvrc_aux.sh
	@ cd ./ilsvrc12 && wget https://github.com/BVLC/caffe/raw/master/python/caffe/imagenet/ilsvrc_2012_mean.npy

unittest:
	@ python -m unittest discover -s tests

#### not used now targets ####
