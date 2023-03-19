.PHONY: help prepare-dev test lint lint-check

VENV_NAME?=venv
VENV_ACTIVATE=. $(VENV_NAME)/bin/activate
PYTHON=${VENV_NAME}/bin/python3.10
export PYTHONPATH := src:$(PYTHONPATH)

.DEFAULT: help
help:
	@echo "make prepare-dev"
	@echo "       prepare development environment, use only once"
	@echo "make test"
	@echo "       run tests"
	@echo "make lint"
	@echo "       run linters"

prepare-dev:
	make venv

venv: $(VENV_NAME)/bin/activate
$(VENV_NAME)/bin/activate: requirements.txt
	test -d $(VENV_NAME) || python3.10 -m venv $(VENV_NAME)
	${PYTHON} -m pip install -U pip pip-tools
	${PYTHON} -m pip install -r requirements.txt
	touch $(VENV_NAME)/bin/activate

test: venv
	$(VENV_ACTIVATE) && ${PYTHON} -m pytest

lint: venv
	$(VENV_ACTIVATE) && ${PYTHON} -m black src/ tests/
	$(VENV_ACTIVATE) && ${PYTHON} -m flake8 src/ tests/ --max-line-length 150 --statistics --show-source


lint-check: venv
	$(VENV_ACTIVATE) && ${PYTHON} -m black --check src/ tests
	$(VENV_ACTIVATE) && ${PYTHON} -m flake8 src/ tests/ --max-line-length 150 --statistics --show-source

requirements: requirements.txt

requirements.txt: requirements.in
	$(VENV_ACTIVATE) && ${VENV_NAME}/bin/pip-compile --output-file=requirements.txt requirements.in

run: venv
	$(VENV_ACTIVATE) && ${PYTHON} src/agent/sim_agent.py /tmp/sim-agent.sock

learn: venv
	$(VENV_ACTIVATE) && ${PYTHON} src/model/learn.py
