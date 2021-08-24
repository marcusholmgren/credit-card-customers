""" Unit test for the churn library """
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
def import_data():
    return cls.import_data


@pytest.fixture()
def perform_eda():
    clear_directory('./images')
    return cls.perform_eda


@pytest.fixture()
def encoder_helper():
    def _func(churn_df, category_lst, response=None):
        cls.encoder_helper(churn_df, category_lst, response)

    return _func


@pytest.fixture()
def perform_feature_engineering():
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    def _func(churn_df, response=None):
        cls.encoder_helper(churn_df, cat_columns, response)
        return cls.perform_feature_engineering(churn_df, response)

    return _func


@pytest.fixture()
def train_models():
    clear_directory('./models')
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    def _func(churn_df):
        cls.encoder_helper(churn_df, cat_columns, response=None)
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(churn_df, response=None)
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


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        churn_df = import_data("./data/bank_data.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
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
    perform_eda(cls.import_data("./data/bank_data.csv"))

    images = list(os.listdir('./images'))
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
    enc_columns = [col + '_Churn' for col in cat_columns]
    churn_df = cls.import_data("./data/bank_data.csv")
    encoder_helper(churn_df, cat_columns, response=None)

    assert set(cat_columns).issubset(churn_df.columns)
    assert set(enc_columns).issubset(churn_df.columns)


def test_perform_feature_engineering(perform_feature_engineering):
    """
    test perform_feature_engineering
    """
    churn_df = cls.import_data("./data/bank_data.csv")
    x_train, x_test, y_train, y_test = perform_feature_engineering(churn_df, response=None)

    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert y_train.shape == (7088,)
    assert y_test.shape == (3039,)
    assert x_train.shape == (7088, 19)
    assert x_test.shape == (3039, 19)


def test_train_models(train_models):
    """
    test train_models
    """
    churn_df = cls.import_data("./data/bank_data.csv")
    train_models(churn_df)

    models = list(os.listdir('./models'))
    models.sort()
    assert len(models) == 2
    assert ['logistic_model.pkl', 'rfc_model.pkl'] == models

    images = list(os.listdir('./images'))
    assert 'feature_importance.png' in images
    assert 'logistic_regression_train.png' in images


if __name__ == "__main__":
    print('To run test suite issue the following command:')
    print('\tpytest -p no:logging -s churn_script_logging_and_tests.py')
