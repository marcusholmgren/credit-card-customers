"""
Customer churn machine learning
"""
# library doc string


# import libraries

from os import PathLike

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def import_data(pth: "PathLike[str]") -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    return pd.read_csv(filepath_or_buffer=pth)


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20, 10))
    ax = df['Churn'].hist()
    ax.set_title('Churning customers')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Attrition')
    plt.savefig('./images/churn_hist.png')

    plt.figure(figsize=(20, 10))
    ax = df['Customer_Age'].hist()
    ax.set_title('Customer Age')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('')
    plt.savefig('./images/customer_age.png')

    plt.figure(figsize=(20, 10))
    ax = df.Marital_Status.value_counts('normalize').plot(kind='bar')
    ax.set_title('Marital Status')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('')
    plt.savefig('./images/marital_status.png')

    plt.figure(figsize=(20, 10))
    ax = sns.distplot(df['Total_Trans_Ct'])
    # ax = sns.displot(df['Total_Trans_Ct'])
    ax.set_title('Total Trans Ct')
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.savefig('./images/total_trans_ct.png')

    plt.figure(figsize=(20, 10))
    ax = sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    ax.set_title('Correlation matrix heatmap')
    # ax.set_ylabel('')
    # ax.set_xlabel('')
    plt.savefig('./images/correlation_heatmap.png')


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    pass


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """


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
    pass


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    pass


def train_models(X_train, X_test, y_train, y_test):
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
    pass
