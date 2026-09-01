# Two-Tesseract SIREN-FWI pipeline.
#
# Both are overridable, so the folder works on another machine or in another
# env without editing this file:
#
#     make run PYTHON=python
#     make build TESSERACT=tesseract
#
# The defaults are this workspace's: geo_jxli is the env with devito, and
# tesseract-core[runtime]/tesseract-jax/equinox were installed into it, so it
# runs the whole in-process pipeline. The tesseract CLI itself lives in the
# torch env. If either default is missing, fall back to whatever is on PATH.
PYTHON    ?= $(shell test -x /home/mtran/miniconda3/envs/geo_jxli/bin/python \
                && echo /home/mtran/miniconda3/envs/geo_jxli/bin/python \
                || echo python)
TESSERACT ?= $(shell test -x /home/mtran/miniconda3/envs/torch/bin/tesseract \
                && echo /home/mtran/miniconda3/envs/torch/bin/tesseract \
                || echo tesseract)

# The CLI shells out to `docker`; this box has rootless podman and no docker
# binary, so builds need the shim at ~/.local/bin/docker.
BUILD_PATH = PATH=$$HOME/.local/bin:$$PATH

.PHONY: help test check-grad run run-full build serve clean

help:
	@echo 'run         the full inversion, reference-script defaults (hours)'
	@echo 'bench       time one loss+gradient and exit'
	@echo 'check-grad  finite-difference the composed gradient'
	@echo 'test        run the checks (mask, both VJPs, composed gradient)'
	@echo 'build       build both container images'
	@echo 'serve       serve both images (prints the URLs for --served)'
	@echo 'clean       remove results/'

# The reference script's defaults: 601x221, 64 shots, SIREN 256x4, omega_0 20,
# adam lr 1.5e-4 with 50 warmup steps, 6000 epochs, and its figure throttles.
# One worker per shot; at ~3 min per loss+gradient this is a multi-day run, so
# it is meant to be started detached (nohup / tmux).
run:
	$(PYTHON) workflow.py \
		--maxiter 6000 --lr 1.5e-4 --warmup 50 \
		--fig-every 25 --wiggle-every 10 --snap-every 5

bench:
	$(PYTHON) workflow.py --benchmark --no-figures

check-grad:
	$(PYTHON) workflow.py --check-grad

test:
	$(PYTHON) test_pipeline.py

build:
	$(BUILD_PATH) $(TESSERACT) build siren_model/
	$(BUILD_PATH) $(TESSERACT) build devito_fwi/

serve:
	$(BUILD_PATH) $(TESSERACT) serve siren-model devito-fwi

clean:
	rm -rf results results_full
