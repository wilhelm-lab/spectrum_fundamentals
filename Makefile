.PHONY: help install lint format type-check test coverage docs docs-serve dist clean check
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:  ## show this help message
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

install:  ## install the package and all dev dependencies
	poetry install

lint:  ## check code style and formatting with ruff
	poetry run ruff check .
	poetry run ruff format --check .

format:  ## auto-fix code style and formatting with ruff
	poetry run ruff format .
	poetry run ruff check --fix .

type-check:  ## run static type checking with mypy
	poetry run mypy spectrum_fundamentals tests

test:  ## run the test suite with pytest
	poetry run pytest

coverage:  ## run tests, combine coverage data, and display report
	poetry run coverage run -m pytest
	poetry run coverage report -i

docs:  ## build HTML documentation with sphinx
	poetry run sphinx-build docs docs/_build/html

docs-serve:  ## build docs and serve locally with live reload
	poetry run sphinx-autobuild docs docs/_build/html --open-browser

dist:  ## build source and wheel packages
	poetry build

clean:  ## remove build, test, and documentation artifacts
	rm -rf dist/ build/ .eggs/
	rm -rf .coverage .coverage.* htmlcov/ .pytest_cache/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name '*.pyc' -delete

check: lint type-check test  ## run all CI checks locally (lint + type-check + test)
