#!/usr/bin/env python3
"""
Unit test for the churn library.

Author: Marcus Holmgren <marcus.holmgren1@gmail.com>
Created: 2021 August
"""
import os
import logging
import shutil
import pytest
import pandas as pd
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@pytest.fixture()
def perform_eda():
    """Test fixture for Exploratory Data Analysis"""
    clear_directory('./images')
    return cls.perform_eda


@pytest.fixture()
def encoder_helper():
    """Test fixture for encoding category fields"""

    def _func(churn_df, category_lst):
        return cls.encoder_helper(churn_df, category_lst)

    return _func


@pytest.fixture()
def perform_feature_engineering():
    """Test fixture for feature engineering"""
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    def _func(churn_df):
        churn_df = cls.encoder_helper(churn_df, cat_columns)
        return cls.perform_feature_engineering(churn_df)

    return _func


@pytest.fixture()
def train_models():
    """Test fixture that trains ML models"""
    clear_directory('./models')
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    def _func(churn_df):
        churn_df = cls.encoder_helper(churn_df, cat_columns)
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(churn_df)
        cls.train_models(x_train, x_test, y_train, y_test)

    return _func


def clear_directory(path):
    """Remove all files or symlinks from the provided path."""
    with os.scandir(path) as entries:
        entry: os.DirEntry
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)


def bank_data() -> pd.DataFrame:
    """ Read bank data into data frame."""
    try:
        churn_df = cls.import_data("./data/bank_data.csv")
        logger.info("Testing bank_data: SUCCESS")
        return churn_df
    except FileNotFoundError as err:
        logger.error("Testing bank_data: The file wasn't found")
        raise err


def test_import():
    """
    test data import
    """
    churn_df = bank_data()

    try:
        assert isinstance(churn_df, pd.DataFrame)
        assert churn_df.shape[0] > 0
        assert churn_df.shape[1] > 0
    except AssertionError as err:
        logger.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    """
    test perform eda function
    """
    assert os.path.exists('./images')
    perform_eda(bank_data())

    images = list(os.listdir('./images/eda'))
    images.sort()
    assert len(images) == 5
    assert ['churn_hist.png', 'correlation_heatmap.png', 'customer_age.png', 'marital_status.png',
            'total_trans_ct.png'] == images


def test_encoder_helper(encoder_helper):
    """
    test encoder helper
    """
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    enc_columns = ['Gender_M', 'Education_Level_Doctorate',
                   'Education_Level_Graduate', 'Education_Level_High School',
                   'Education_Level_Post-Graduate', 'Education_Level_Uneducated',
                   'Education_Level_Unknown', 'Marital_Status_Married',
                   'Marital_Status_Single', 'Marital_Status_Unknown',
                   'Income_Category_$40K - $60K', 'Income_Category_$60K - $80K',
                   'Income_Category_$80K - $120K', 'Income_Category_Less than $40K',
                   'Income_Category_Unknown', 'Card_Category_Gold',
                   'Card_Category_Platinum', 'Card_Category_Silver']
    churn_df = bank_data()
    churn_df = encoder_helper(churn_df, cat_columns)

    assert not set(cat_columns).issubset(churn_df.columns)
    assert set(enc_columns).issubset(churn_df.columns)


def test_perform_feature_engineering(perform_feature_engineering):
    """
    test perform_feature_engineering
    """
    churn_df = bank_data()
    x_train, x_test, y_train, y_test = perform_feature_engineering(churn_df)

    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert y_train.shape == (7088,)
    assert y_test.shape == (3039,)
    assert x_train.shape == (7088, 32)
    assert x_test.shape == (3039, 32)


def test_train_models(train_models):
    """
    test train_models
    """
    churn_df = bank_data()
    train_models(churn_df)

    models = list(os.listdir('./models'))
    models.sort()
    assert len(models) == 2
    assert ['logistic_model.pkl', 'rfc_model.pkl'] == models

    images = list(os.listdir('./images/results'))
    assert 'feature_importance.png' in images
    assert 'logistic_regression_train.png' in images
    assert 'random_forest_train.png' in images


if __name__ == "__main__":
    print('To run test suite issue the following command:')
    print('\tpytest -p no:logging -s churn_script_logging_and_tests.py')
    print('Or if your system have GNU make use command:')
    print('\tmake test')
