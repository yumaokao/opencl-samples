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
	$(call mkdir_config, models/bvlc_alexnet)
	@ python ../tools/download_model_binary.py models/bvlc_alexnet

unittest:
	@ python -m unittest discover -s tests

#### not used now targets ####
