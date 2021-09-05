.DEFAULT_GOAL := help

help:
		@echo "Please use 'make <target>' where <target> is one of"
		@echo ""
		@echo "  setup       create python virtual environment"
		@echo "  install     install packages"
		@echo "  format      reformat code"
		@echo "  lint        lint code"
		@echo "  test        run all the tests"
		@echo "  clean       remove *.pyc files and __pycache__ directory"
		@echo ""
		@echo "Check the Makefile to know exactly what each target is doing."

setup:
	## Recommend you create a virtualenv
	python3 -m venv venv
	source venv/bin/activate

install:
	pip install --upgrade pip && \
		pip install -r requirements.txt

format:
	autopep8 --in-place --aggressive --aggressive churn_library.py

lint:
	pylint churn_library.py churn_script_logging_and_tests.py

test:
	pytest -vv -p no:logging -s churn_script_logging_and_tests.py

clean:
	find . -type d -name "__pycache__" | xargs rm -rf {};
	rm -rf venv
