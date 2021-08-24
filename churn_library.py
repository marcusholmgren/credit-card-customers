"""
Customer churn machine learning library.

Functionality to perform feature engineering, exploratory data analysis and model training.
"""
import logging
from os import PathLike
from typing import Tuple

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


def import_data(pth: "PathLike[str]") -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    import_df = pd.read_csv(filepath_or_buffer=pth)
    import_df['Churn'] = import_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return import_df


def perform_eda(churn_df: pd.DataFrame):
    """
    perform eda on dataframe and save figures to images folder
    input:
            churn_df: pandas dataframe

    output:
            None
    """
    plt.figure(figsize=(20, 10))
    axes = churn_df['Churn'].hist()
    axes.set_title('Churning customers')
    axes.set_ylabel('Frequency')
    axes.set_xlabel('Attrition')
    plt.savefig('./images/churn_hist.png')

    plt.figure(figsize=(20, 10))
    axes = churn_df['Customer_Age'].hist()
    axes.set_title('Customer Age')
    axes.set_ylabel('Frequency')
    axes.set_xlabel('Age')
    plt.tight_layout()
    plt.savefig('./images/customer_age.png')

    plt.figure(figsize=(20, 10))
    axes = churn_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    axes.set_title('Marital Status')
    axes.set_ylabel('Frequency')
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('./images/marital_status.png')

    plt.figure(figsize=(20, 10))
    cfg = sns.displot(churn_df, x='Total_Trans_Ct', kde=True)
    cfg.set(title='Total Trans Ct')
    plt.tight_layout()
    plt.savefig('./images/total_trans_ct.png')

    plt.figure(figsize=(20, 10))
    axes = sns.heatmap(
        churn_df.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    axes.set_title('Correlation matrix heatmap')
    plt.tight_layout()
    plt.savefig('./images/correlation_heatmap.png')


def encoder_helper(
        churn_df: pd.DataFrame,
        category_lst: "list[str]",
        response: "list[str]"):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            churn_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                                              for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """

    def _calc_mean_churn(column: str):
        groups: pd.Series = churn_df.groupby(column).mean()['Churn']
        lst = [groups.loc[val] for val in churn_df[column]]
        churn_df[f'{column}_Churn'] = lst

    for category in category_lst:
        _calc_mean_churn(category)


def perform_feature_engineering(churn_df: pd.DataFrame,
                                response: "list[str]") \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    input:
              churn_df: pandas dataframe
              response: string of response name [optional argument that could be used
                                                for naming variables or index y column]

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
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
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
        plt.savefig(f'./images/{name.replace(" ", "_").lower()}_train.png')

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
                            output_pth: "PathLike[str]"):
    """
    creates and stores the feature importances in pth
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
    plt.xticks(range(features_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
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
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    logger.info('random forest results')
    logger.info('test results')
    logger.info(classification_report(y_test, y_test_preds_rf))
    logger.info('train results')
    logger.info(classification_report(y_train, y_train_preds_rf))

    logger.info('logistic regression results')
    logger.info('test results')
    logger.info(classification_report(y_test, y_test_preds_lr))
    logger.info('train results')
    logger.info(classification_report(y_train, y_train_preds_lr))

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    all_features = pd.concat([X_train, X_test], axis=0)
    feature_importance_plot(
        cv_rfc,
        all_features,
        './images/feature_importance.png')
