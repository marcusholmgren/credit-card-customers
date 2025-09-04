.DEFAULT_GOAL := help

help:
		@echo "Please use 'make <target>' where <target> is one of"
		@echo ""
		@echo "  install     install packages"
		@echo "  format      reformat code"
		@echo "  lint        lint code"
		@echo "  test        run all the tests"
		@echo "  clean       remove *.pyc files and __pycache__ directory"
		@echo ""
		@echo "Check the Makefile to know exactly what each target is doing."

install:
	uv sync

format:
	uv run ruff format

lint:
	uv run ruff check

test:
	pytest -vv -p no:logging -s churn_script_logging_and_tests.py

clean:
	find . -type d -name "__pycache__" | xargs rm -rf {};
	rm -rf venv
