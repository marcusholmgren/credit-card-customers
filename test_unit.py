"""
Unit tests for the churn library functions.
These tests focus on individual function logic using mock data.

Author: Jules
Created: 2024
"""

import os
import shutil
import pytest
import pandas as pd
import numpy as np
from io import StringIO
import churn_library as cls


@pytest.fixture(scope="module")
def config():
    """Fixture to load the project configuration."""
    return cls.load_config()


@pytest.fixture(scope="function")
def temp_output_dirs(config):
    """Fixture to create temporary directories for test outputs and clean them up."""
    dirs_to_create = [
        config["eda"]["image_dir"],
        config["models"]["results"]["image_dir"],
        config["model_dir"],
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

    yield  # This allows the test to run

    # Teardown: remove created directories
    for d in dirs_to_create:
        if os.path.exists(d):
            shutil.rmtree(d)


def test_import_data():
    """Test the import_data function for correct DataFrame creation."""
    csv_data = """
Unnamed: 0,CLIENTNUM,Attrition_Flag,Customer_Age,Gender
1,123,Existing Customer,45,M
2,456,Attrited Customer,50,F
"""
    # Use StringIO to simulate a file
    csv_file = StringIO(csv_data)

    df = cls.import_data(csv_file)

    assert isinstance(df, pd.DataFrame)
    assert "Churn" in df.columns
    assert "Unnamed: 0" not in df.columns
    assert "CLIENTNUM" not in df.columns
    assert df["Churn"].tolist() == [0, 1]
    assert df.shape[0] == 2  # 2 rows of data
    assert df.shape[1] == 3  # Customer_Age, Gender, Churn


def test_get_preprocessor(config):
    """Test that the preprocessor is configured correctly."""
    preprocessor = cls.get_preprocessor()

    numeric_features = config["feature_engineering"]["numeric_features"]
    cat_features = config["feature_engineering"]["categorical_features"]

    # Check that the transformers target the correct columns
    assert preprocessor.transformers[0][2] == numeric_features
    assert preprocessor.transformers[1][2] == cat_features


def test_classification_report_image(temp_output_dirs, config):
    """Test that the classification report image is generated."""
    y_train = pd.Series([0, 1, 0, 1])
    y_test = pd.Series([0, 1, 0, 1])
    y_train_preds = pd.Series([0, 0, 1, 1])
    y_test_preds = pd.Series([0, 1, 0, 0])

    report_path = os.path.join(
        config["models"]["results"]["image_dir"], "test_model_report.png"
    )

    cls.classification_report_image(
        y_train, y_test, y_train_preds, y_test_preds, "Test Model"
    )

    # Check if the file was created
    assert os.path.exists(report_path)


def test_feature_importance_plot(temp_output_dirs, config):
    """Test that the feature importance plot is generated."""

    # Create a mock model and data
    class MockModel:
        def __init__(self):
            self.feature_importances_ = np.array([0.1, 0.3, 0.6])

    mock_model = MockModel()
    feature_names = ["feat_a", "feat_b", "feat_c"]
    output_path = os.path.join(
        config["models"]["results"]["image_dir"], "test_feature_importance.png"
    )

    cls.feature_importance_plot(mock_model, feature_names, output_path)

    assert os.path.exists(output_path)
