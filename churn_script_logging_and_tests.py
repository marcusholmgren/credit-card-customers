#!/usr/bin/env python3
"""
Unit test for the churn library.

Author: Marcus Holmgren <marcus.holmgren1@gmail.com>
Created: 2021 August
Refactored: 2024
"""
import os
import logging
import shutil
import pytest
import pandas as pd
import yaml
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import churn_library as cls

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Configure logging
log_dir = config.get('log_dir', './logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'churn_library.log'),
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@pytest.fixture(scope='module')
def bank_data() -> pd.DataFrame:
    """Fixture to load the bank data."""
    try:
        df = cls.import_data(config['data_path'])
        logger.info("Fixture bank_data: SUCCESS")
        return df
    except FileNotFoundError as err:
        logger.error("Fixture bank_data: The file wasn't found")
        raise err

def test_import(bank_data):
    """Test data import."""
    try:
        assert isinstance(bank_data, pd.DataFrame)
        assert bank_data.shape[0] > 0
        assert bank_data.shape[1] > 0
        assert 'Churn' in bank_data.columns
        assert 'Unnamed: 0' not in bank_data.columns # Check that unnecessary columns are dropped
        logger.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logger.error("Testing import_data: Failed assertions")
        raise err

def test_perform_eda(bank_data):
    """Test perform_eda function."""
    eda_dir = config['eda']['image_dir']
    if os.path.exists(eda_dir):
        shutil.rmtree(eda_dir)

    cls.perform_eda(bank_data)

    assert os.path.exists(eda_dir)

    expected_files = [plot['filename'] for plot in config['eda']['plots'].values()]
    generated_files = os.listdir(eda_dir)

    assert len(generated_files) == len(expected_files)
    assert sorted(generated_files) == sorted(expected_files)
    logger.info("Testing perform_eda: SUCCESS")

def test_get_preprocessor():
    """Test the get_preprocessor function."""
    preprocessor = cls.get_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)
    # Check that it has transformers for numeric and categorical features
    assert len(preprocessor.transformers) == 2
    assert preprocessor.transformers[0][0] == 'num'
    assert preprocessor.transformers[1][0] == 'cat'
    logger.info("Testing get_preprocessor: SUCCESS")

def test_perform_feature_engineering(bank_data):
    """Test perform_feature_engineering function."""
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(bank_data)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    assert X_train.shape[0] + X_test.shape[0] == bank_data.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == bank_data.shape[0]

    logger.info("Testing perform_feature_engineering: SUCCESS")

def test_train_models(bank_data):
    """Test train_models function."""
    model_dir = config['model_dir']
    results_dir = config['models']['results']['image_dir']

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(bank_data)
    cls.train_models(X_train, X_test, y_train, y_test)

    # Check if models are saved
    assert os.path.exists(config['models']['logistic_regression']['model_path'])
    assert os.path.exists(config['models']['random_forest']['model_path'])
    assert os.path.exists(config['models']['lightgbm']['model_path'])

    # Check if result images are saved
    assert os.path.exists(os.path.join(results_dir, 'logistic_regression_report.png'))
    assert os.path.exists(os.path.join(results_dir, 'random_forest_report.png'))
    assert os.path.exists(os.path.join(results_dir, 'lightgbm_report.png'))

    # Check for feature importance plots
    assert os.path.exists(os.path.join(results_dir, 'random_forest_feature_importance.png'))
    assert os.path.exists(os.path.join(results_dir, 'lightgbm_feature_importance.png'))

    # Test loading a model and making a prediction
    rfc_model: Pipeline = joblib.load(config['models']['random_forest']['model_path'])
    prediction = rfc_model.predict(X_test.iloc[0:1])
    assert prediction is not None

    logger.info("Testing train_models: SUCCESS")

if __name__ == "__main__":
    print("Running tests...")
    pytest.main(['-s', __file__])
