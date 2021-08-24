# Credit Card customers

Predict Churning customers from a Credit Card dataset

* [Churn notebook](churn_notebook.ipynb) - raw Jupyter notebook with code for EDA and machine learning model selection.
* [churn_library.py](churn_library.py) - extracted code from the Jupyter notebook
* [churn_script_logging_and_tests.py](churn_script_logging_and_tests.py) - unit test for the `churn_library`

Folder:
* __data__ - the raw data file for the analysis and model creation.
* __images__ - contains all the images produced from EDA and model selection.
* __logs__ - log file produced from running the `churn_library`
* __models__ - pickled trained model

## Run the code

Recommended to create a virtual environment to install dependencies
```
python3 -m venv venv
```

Activate the virtual environment
```
source venv/bin/activate
```

Installs the third party requirements for running the churn analysis and unit tests
```
pip install -r requirements.txt
```

Runs all the tests in the test suite
```
pytest -p no:logging -s churn_script_logging_and_tests.py
```

Lint the `churn_library.py`
```
pylint churn_library.py
```

Format the `churn_library.py` according to [PEP-8](https://www.python.org/dev/peps/pep-0008/) rules
```
autopep8 --in-place --aggressive --aggressive churn_library.py 
```

### MAKE

`Makefile` is a [GNU make](https://www.gnu.org/software/make/manual/make.html) script that allows for running the `lint`, `format` and `test` commands.

Run `make` command from CLI to see available options.

Runs the make lint command
```
make lint
```

## Tech Stack

**Server:** Python, Jupyter, Pandas, Matplotlib, Seaborn, NumPy, scikit-learn, pytest

## Resources

* [Udacity](https://www.udacity.com/)
* [Machine Learning DevOps Engineer](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821)
* [README](https://readme.so/) template