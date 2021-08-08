import os
import pytest
import logging
import shutil
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
    return lambda pth: cls.import_data(pth)


@pytest.fixture()
def perform_eda():
    clear_directory('./images')
    return lambda df: cls.perform_eda(df)


@pytest.fixture()
def encoder_helper():
    def _my(df, category_lst, response=None):
        cls.encoder_helper(df, category_lst, response)
    return _my


@pytest.fixture()
def perform_feature_engineering():
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    def _my(df, response=None):
        cls.encoder_helper(df, cat_columns, response)
        return cls.perform_feature_engineering(df, response)
    return _my


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

    def _my(df):
        cls.encoder_helper(df, cat_columns, response=None)
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, response=None)
        cls.train_models(X_train, X_test, y_train, y_test)
    return _my


def clear_directory(path):
    with os.scandir(path) as entries:
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
        df = import_data("./data/bank_data.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logger.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    """
    test perform eda function
    """
    assert os.path.exists('./images')
    perform_eda(cls.import_data("./data/bank_data.csv"))

    images = [img for img in os.listdir('./images')]
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
    df = cls.import_data("./data/bank_data.csv")
    encoder_helper(df, cat_columns, response=None)

    assert set(cat_columns).issubset(df.columns)
    assert set(enc_columns).issubset(df.columns)


def test_perform_feature_engineering(perform_feature_engineering):
    """
    test perform_feature_engineering
    """
    df = cls.import_data("./data/bank_data.csv")
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, response=None)

    assert type(X_train) is pd.DataFrame
    assert type(X_test) is pd.DataFrame
    assert type(y_train) is pd.Series
    assert type(y_test) is pd.Series
    assert y_train.shape == (7088,)
    assert y_test.shape == (3039,)
    assert X_train.shape == (7088, 19)
    assert X_test.shape == (3039, 19)


def test_train_models(train_models):
    """
    test train_models
    """
    df = cls.import_data("./data/bank_data.csv")
    train_models(df)

    models = [mdl for mdl in os.listdir('./models')]
    models.sort()
    assert len(models) == 2
    assert ['logistic_model.pkl', 'rfc_model.pkl'] == models


if __name__ == "__main__":
    print('To run test suite issue the following command:')
    print('\tpytest -p no:logging -s churn_script_logging_and_tests.py')
    # churn_df = cls.import_data("./data/bank_data.csv")
    # test_import(cls.import_data)
    # test_eda(lambda x: cls.perform_eda(churn_df))
    # test_encoder_helper(lambda x, y, response: cls.encoder_helper(churn_df, y, response))
    # test_perform_feature_engineering(lambda x, response: cls.perform_feature_engineering(churn_df, response))
