import os
import pytest
import logging
import shutil
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture()
def import_data():
    return lambda pth: cls.import_data(pth)


@pytest.fixture()
def perform_eda():
    with os.scandir('./images') as entries:
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
    return lambda df: cls.perform_eda(df)


@pytest.fixture()
def encoder_helper():
    pass


@pytest.fixture()
def perform_feature_engineering():
    pass


@pytest.fixture()
def train_models():
    pass


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
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
    pass


def test_perform_feature_engineering(perform_feature_engineering):
    """
    test perform_feature_engineering
    """
    pass


def test_train_models(train_models):
    """
    test train_models
    """
    pass


if __name__ == "__main__":
    churn_df = cls.import_data("./data/bank_data.csv")
    test_import(cls.import_data)
    test_eda(cls.perform_eda(churn_df))
