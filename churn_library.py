"""
Customer churn machine learning library.

Functionality to perform feature engineering, exploratory data analysis and model training.

Author: Marcus Holmgren <marcus.holmgren1@gmail.com>
Created: 2021 August
"""
import logging
import os
from os import PathLike
from typing import Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

sns.set()
logger = logging.getLogger(__name__)


def import_data(pth: Union[str, "PathLike[str]"]) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    create_dir('./logs')
    import_df = pd.read_csv(filepath_or_buffer=pth)
    import_df['Churn'] = pd.get_dummies(
        import_df['Attrition_Flag'])['Attrited Customer']
    return import_df


def perform_eda(churn_df: pd.DataFrame):
    """
    perform eda on dataframe and save figures to images folder
    input:
            churn_df: pandas dataframe

    output:
            None
    """
    create_dir('./images/eda')
    plt.figure(figsize=(20, 10))
    axes = churn_df['Churn'].hist()
    axes.set_title('Churning customers')
    axes.set_ylabel('Frequency')
    axes.set_xlabel('Attrition')
    plt.savefig('./images/eda/churn_hist.png')

    plt.figure(figsize=(20, 10))
    axes = churn_df['Customer_Age'].hist()
    axes.set_title('Customer Age')
    axes.set_ylabel('Frequency')
    axes.set_xlabel('Age')
    plt.tight_layout()
    plt.savefig('./images/eda/customer_age.png')

    plt.figure(figsize=(20, 10))
    axes = churn_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    axes.set_title('Marital Status')
    axes.set_ylabel('Frequency')
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('./images/eda/marital_status.png')

    plt.figure(figsize=(20, 10))
    cfg = sns.displot(churn_df, x='Total_Trans_Ct', kde=True)
    cfg.set(title='Total Trans Ct')
    plt.tight_layout()
    plt.savefig('./images/eda/total_trans_ct.png')

    plt.figure(figsize=(20, 10))
    axes = sns.heatmap(
        churn_df.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    axes.set_title('Correlation matrix heatmap')
    plt.tight_layout()
    plt.savefig('./images/eda/correlation_heatmap.png')


def encoder_helper(
        churn_df: pd.DataFrame,
        category_lst: "list[str]"):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            churn_df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            df: pandas dataframe with new columns for
    """
    return pd.get_dummies(churn_df, columns=category_lst, drop_first=True)


def perform_feature_engineering(churn_df: pd.DataFrame) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    input:
              churn_df: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    target = churn_df['Churn']
    features = pd.DataFrame()
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_M', 'Education_Level_Doctorate',
        'Education_Level_Graduate', 'Education_Level_High School',
        'Education_Level_Post-Graduate', 'Education_Level_Uneducated',
        'Education_Level_Unknown', 'Marital_Status_Married',
        'Marital_Status_Single', 'Marital_Status_Unknown',
        'Income_Category_$40K - $60K', 'Income_Category_$60K - $80K',
        'Income_Category_$80K - $120K', 'Income_Category_Less than $40K',
        'Income_Category_Unknown', 'Card_Category_Gold',
        'Card_Category_Platinum', 'Card_Category_Silver']
    features[keep_cols] = churn_df[keep_cols]

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """

    def _classification_fig(name: str, target_train,
                            target_test,
                            target_train_preds,
                            target_test_preds):
        plt.figure(figsize=(7, 7))
        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str(f'{name} Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(target_test, target_test_preds)),
                 {'fontsize': 10},
                 fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str(f'{name} Test'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(target_train, target_train_preds)),
                 {'fontsize': 10},
                 fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        plt.savefig(
            f'./images/results/{name.replace(" ", "_").lower()}_train.png')

    _classification_fig(
        'Random Forest',
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf)
    _classification_fig(
        'Logistic Regression',
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr)


def feature_importance_plot(model: GridSearchCV,
                            features_data: pd.DataFrame,
                            output_pth: Union[str, "PathLike[str]"]):
    """
    creates and stores the feature importance's in pth
    input:
            model: model object containing feature_importances_
            features_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [features_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(features_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(features_data.shape[1]), names, rotation=45)
    plt.tight_layout()
    plt.savefig(output_pth)


def train_models(
        features_train: pd.DataFrame,
        features_test: pd.DataFrame,
        target_train: pd.Series,
        target_test: pd.Series):
    """
    train, store model results: images + scores, and store models
    input:
              features_train: X training data
              features_test: X testing data
              target_train: y training data
              target_test: y testing data
    output:
              None
    """
    create_dir('./images/results')
    create_dir('./models')
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(features_train, target_train)

    lrc.fit(features_train, target_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(features_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(features_test)

    y_train_preds_lr = lrc.predict(features_train)
    y_test_preds_lr = lrc.predict(features_test)

    # scores
    logger.info('random forest results')
    logger.info('test results')
    logger.info(classification_report(target_test, y_test_preds_rf))
    logger.info('train results')
    logger.info(classification_report(target_train, y_train_preds_rf))

    logger.info('logistic regression results')
    logger.info('test results')
    logger.info(classification_report(target_test, y_test_preds_lr))
    logger.info('train results')
    logger.info(classification_report(target_train, y_train_preds_lr))

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    classification_report_image(
        target_train,
        target_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    all_features = pd.concat([features_train, features_test], axis=0)
    feature_importance_plot(
        cv_rfc,
        all_features,
        './images/results/feature_importance.png')


def create_dir(path: Union[str, "PathLike[str]"]):
    """ Create target Directory if don't exist.
    input:
            path: directory to create
    output:
            None
    """
    if not os.path.exists(path):
        os.mkdir(path)
        logger.info('Directory %s Created.', path)
    else:
        logger.info('Directory %s already exists.', path)
