# Licensed under the MIT License
# https://github.com/craigahobbs/picoocr/blob/main/LICENSE


# Download javascript-build
JAVASCRIPT_BUILD_DIR ?= ../javascript-build
define WGET
ifeq '$$(wildcard $(notdir $(1)))' ''
$$(info Downloading $(notdir $(1)))
$$(shell [ -f $(JAVASCRIPT_BUILD_DIR)/$(notdir $(1)) ] && cp $(JAVASCRIPT_BUILD_DIR)/$(notdir $(1)) . || $(call WGET_CMD, $(1)))
endif
endef
WGET_CMD = if command -v wget >/dev/null 2>&1; then wget -q -c $(1); else curl -f -Os $(1); fi
$(eval $(call WGET, https://craigahobbs.github.io/javascript-build/Makefile.base))
$(eval $(call WGET, https://craigahobbs.github.io/javascript-build/jsdoc.json))
$(eval $(call WGET, https://craigahobbs.github.io/javascript-build/eslint.config.js))


# Include javascript-build
include Makefile.base


# The development tools are linted too; the generated model module (lib/models/) isn't
ESLINT_ARGS := $(ESLINT_ARGS) tools/ --ignore-pattern 'lib/models/**'


clean:
	rm -rf Makefile.base jsdoc.json eslint.config.js


help:
	@echo '            [commit-train|parity|train|train-export|train-float|train-install]'


# The training package (train/ - a python-build package, picoocr_train): make commit runs its gate too (its unit tests,
# pylint, and 100% branch coverage), and make clean cleans it
.PHONY: commit-train
commit: commit-train
commit-train:
	$(MAKE) -C train commit

clean: clean-train
.PHONY: clean-train
clean-train:
	$(MAKE) -C train clean


# The Python environment for make train - build/venv-train from train/requirements.txt (PyTorch's CPU wheels on Linux
# by TORCH_INDEX - PyPI's Linux wheels bundle CUDA). PYTHON_VENV is the Python that creates it.
PYTHON_VENV ?= python3
TORCH_INDEX ?= $(if $(filter Linux,$(shell uname -s)),--extra-index-url https://download.pytorch.org/whl/cpu)

build/venv-train.build: train/requirements.txt
	rm -rf build/venv-train
	$(PYTHON_VENV) -m venv build/venv-train
	build/venv-train/bin/pip -q install --upgrade pip
	build/venv-train/bin/pip -q install $(TORCH_INDEX) -r train/requirements.txt
	touch $@


# make train - the recognizer from the recipe (train/data/train.json) in TRAIN_DIR/run, its fonts and books fetched
# (downloads/); then make train-export - its best checkpoint as TRAIN_DIR/recognizer.js, checked against the package
# (make parity); then make train-install - the trained module into lib/models/
TRAIN_DIR ?= build/train
TRAIN_ARGS ?=
TRAIN_PYTHON := env PYTHONPATH=$(CURDIR)/train/src build/venv-train/bin/python -B

.PHONY: train train-export train-install parity
train: build/venv-train.build
	$(TRAIN_PYTHON) -m picoocr_train.fonts check
	$(TRAIN_PYTHON) -m picoocr_train.train $(TRAIN_DIR)/run $(TRAIN_ARGS)

train-export: build/venv-train.build
	$(TRAIN_PYTHON) -m picoocr_train.export $(TRAIN_DIR)/run/best.pt -o $(TRAIN_DIR)/recognizer.js --parity $(TRAIN_DIR)/parity.json
	$(MAKE) parity

parity:
	$(NODE_SHELL) node tools/parity.js $(TRAIN_DIR)/recognizer.js $(TRAIN_DIR)/parity.json

train-install:
	cp $(TRAIN_DIR)/recognizer.js lib/models/recognizer.js

# make train-float - the float32 export (TRAIN_DIR/float.js), the quantized model's reference: node tools/pagecheck.js
# PAGES --model TRAIN_DIR/recognizer.js --reference TRAIN_DIR/float.js
.PHONY: train-float
train-float: build/venv-train.build
	$(TRAIN_PYTHON) -m picoocr_train.export $(TRAIN_DIR)/run/best.pt -o $(TRAIN_DIR)/float.js --float
