"""
Customer churn machine learning library.
Functionality to perform feature engineering, exploratory data analysis, and model training.

Author: Marcus Holmgren <marcus.holmgren1@gmail.com>
Created: 2021 August
Refactored: 2024
"""

import logging
import os
from os import PathLike
from typing import Tuple, Union
import mlflow
import yaml

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config()


def import_data(pth: Union[str, "PathLike[str]"]) -> pd.DataFrame:
    """
    Returns a DataFrame for the CSV found at pth.
    input:
        pth: a path to the csv
    output:
        df: pandas dataframe
    """
    os.makedirs(config["log_dir"], exist_ok=True)
    df = pd.read_csv(filepath_or_buffer=pth)
    df = df.drop(columns=["Unnamed: 0", "CLIENTNUM"])
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    df = df.drop(columns=["Attrition_Flag"])
    return df


def _save_plot(filename: str, directory: str):
    """Helper function to save plots."""
    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory, filename))
    plt.close()


def plot_churn_histogram(df: pd.DataFrame):
    """Plot and save churn histogram."""
    plt.figure(figsize=(20, 10))
    plot_config = config["eda"]["plots"]["churn_hist"]
    axes = df["Churn"].hist()
    axes.set_title(plot_config["title"])
    axes.set_ylabel(plot_config["ylabel"])
    axes.set_xlabel(plot_config["xlabel"])
    _save_plot(plot_config["filename"], config["eda"]["image_dir"])


def plot_customer_age_histogram(df: pd.DataFrame):
    """Plot and save customer age histogram."""
    plt.figure(figsize=(20, 10))
    plot_config = config["eda"]["plots"]["customer_age"]
    axes = df["Customer_Age"].hist()
    axes.set_title(plot_config["title"])
    axes.set_ylabel(plot_config["ylabel"])
    axes.set_xlabel(plot_config["xlabel"])
    plt.tight_layout()
    _save_plot(plot_config["filename"], config["eda"]["image_dir"])


def plot_marital_status_bar(df: pd.DataFrame):
    """Plot and save marital status bar chart."""
    plt.figure(figsize=(20, 10))
    plot_config = config["eda"]["plots"]["marital_status"]
    axes = df.Marital_Status.value_counts("normalize").plot(kind="bar")
    axes.set_title(plot_config["title"])
    axes.set_ylabel(plot_config["ylabel"])
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    _save_plot(plot_config["filename"], config["eda"]["image_dir"])


def plot_total_trans_ct_distplot(df: pd.DataFrame):
    """Plot and save total transaction count distribution."""
    plt.figure(figsize=(20, 10))
    plot_config = config["eda"]["plots"]["total_trans_ct"]
    cfg = sns.histplot(data=df, x="Total_Trans_Ct", stat="density", kde=True)
    cfg.set_title(plot_config["title"])
    plt.tight_layout()
    _save_plot(plot_config["filename"], config["eda"]["image_dir"])


def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot and save correlation heatmap."""
    plt.figure(figsize=(20, 10))
    plot_config = config["eda"]["plots"]["correlation_heatmap"]
    numeric_df = df.select_dtypes(include="number")
    axes = sns.heatmap(numeric_df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    axes.set_title(plot_config["title"])
    plt.tight_layout()
    _save_plot(plot_config["filename"], config["eda"]["image_dir"])


def perform_eda(df: pd.DataFrame):
    """
    Perform EDA on the dataframe and save figures to the images folder.
    """
    plot_churn_histogram(df)
    plot_customer_age_histogram(df)
    plot_marital_status_bar(df)
    plot_total_trans_ct_distplot(df)
    plot_correlation_heatmap(df)


def perform_feature_engineering(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data and defines the preprocessing pipeline.
    """
    target = df[config["feature_engineering"]["target_col"]]
    features = df.drop(columns=[config["feature_engineering"]["target_col"]])

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config["feature_engineering"]["test_size"],
        random_state=config["feature_engineering"]["random_state"],
        stratify=target,
    )
    return X_train, X_test, y_train, y_test


def get_preprocessor() -> ColumnTransformer:
    """Returns a ColumnTransformer for preprocessing features."""
    cat_features = config["feature_engineering"]["categorical_features"]
    numeric_features = config["feature_engineering"]["numeric_features"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="passthrough",
    )
    return preprocessor


def classification_report_image(
    y_train, y_test, y_train_preds, y_test_preds, model_name: str
):
    """
    Produces a classification report for training and testing results and stores the report as an image.
    """
    plt.figure(figsize=(7, 7))
    plt.rc("figure", figsize=(5, 5))
    plt.text(
        0.01,
        1.25,
        str(f"{model_name} Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.6,
        str(classification_report(y_train, y_train_preds)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.5,
        str(f"{model_name} Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        -0.1,
        str(classification_report(y_test, y_test_preds)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.axis("off")

    results_dir = config["models"]["results"]["image_dir"]
    _save_plot(f"{model_name.replace(' ', '_').lower()}_report.png", results_dir)


def feature_importance_plot(model, feature_names: list, output_pth: str):
    """
    Creates and stores the feature importance plot.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [feature_names[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(len(names)), importances[indices])
    plt.xticks(range(len(names)), names, rotation=90)
    plt.tight_layout()
    _save_plot(os.path.basename(output_pth), os.path.dirname(output_pth))


def train_models(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
):
    """
    Train, store model results, and save models.
    """
    os.makedirs(config["model_dir"], exist_ok=True)
    preprocessor = get_preprocessor()

    # Define models
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=config["feature_engineering"]["random_state"],
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        random_state=config["feature_engineering"]["random_state"]
                    ),
                ),
            ]
        ),
        "LightGBM": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    lgb.LGBMClassifier(
                        random_state=config["feature_engineering"]["random_state"]
                    ),
                ),
            ]
        ),
    }

    mlflow.set_experiment("churn_prediction")
    # Train and tune models
    for name, model_pipeline in models.items():
        logger.info(f"Training {name}...")
        with mlflow.start_run(run_name=name):
            if name in ["Random Forest", "LightGBM"]:
                param_dist = config["models"][name.lower().replace(" ", "_")][
                    "param_dist"
                ]
                # Prefix parameters with 'classifier__' for pipeline
                param_dist_prefixed = {
                    f"classifier__{k}": v for k, v in param_dist.items()
                }

                search = RandomizedSearchCV(
                    model_pipeline,
                    param_distributions=param_dist_prefixed,
                    n_iter=config["models"]["n_iter_search"],
                    cv=5,
                    random_state=config["feature_engineering"]["random_state"],
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                mlflow.log_params(param_dist)
                mlflow.log_params(search.best_params_)
            else:
                model_pipeline.fit(X_train, y_train)
                best_model = model_pipeline

            # Predictions
            y_train_preds = best_model.predict(X_train)
            y_test_preds = best_model.predict(X_test)

            # Logging results
            logger.info(
                f"{name} Results:\n{classification_report(y_test, y_test_preds)}"
            )
            report = classification_report(y_test, y_test_preds, output_dict=True)

            # Flatten the report for MLflow logging
            report_flat = {}
            for key, value in report.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        report_flat[f"{key}_{sub_key}"] = sub_value
                else:
                    report_flat[key] = value
            mlflow.log_metrics(report_flat)

            # Save model
            model_path_key = name.lower().replace(" ", "_")
            joblib.dump(best_model, config["models"][model_path_key]["model_path"])
            mlflow.sklearn.log_model(
                sk_model=best_model,
                name=model_path_key,
                input_example=X_train.head(),
            )

            # Generate and save classification reports as images
            classification_report_image(
                y_train, y_test, y_train_preds, y_test_preds, name
            )
            report_image_path = os.path.join(
                config["models"]["results"]["image_dir"],
                f"{name.replace(' ', '_').lower()}_report.png",
            )
            mlflow.log_artifact(report_image_path)

            # Feature importance for tree-based models
            if name in ["Random Forest", "LightGBM"]:
                try:
                    feature_names = best_model.named_steps[
                        "preprocessor"
                    ].get_feature_names_out()
                    feature_importance_path = os.path.join(
                        config["models"]["results"]["image_dir"],
                        f"{model_path_key}_feature_importance.png",
                    )
                    feature_importance_plot(
                        best_model.named_steps["classifier"],
                        list(feature_names),
                        feature_importance_path,
                    )
                    mlflow.log_artifact(feature_importance_path)
                except Exception as e:
                    logger.error(
                        f"Could not generate feature importance plot for {name}: {e}"
                    )



if __name__ == "__main__":
    df = import_data(config["data_path"])
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    train_models(X_train, X_test, y_train, y_test)
