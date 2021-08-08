"""
Customer churn machine learning
"""
# library doc string


# import libraries

from os import PathLike

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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


def perform_eda(df: pd.DataFrame):
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
    ax.set_xlabel('Age')
    plt.tight_layout()
    plt.savefig('./images/customer_age.png')

    plt.figure(figsize=(20, 10))
    ax = df.Marital_Status.value_counts('normalize').plot(kind='bar')
    ax.set_title('Marital Status')
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('./images/marital_status.png')

    plt.figure(figsize=(20, 10))
    #ax = sns.distplot(df['Total_Trans_Ct'])
    cfg = sns.displot(df, x='Total_Trans_Ct', kde=True)
    cfg.set(title='Total Trans Ct')
    plt.tight_layout()
    plt.savefig('./images/total_trans_ct.png')

    plt.figure(figsize=(20, 10))
    ax = sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    ax.set_title('Correlation matrix heatmap')
    plt.tight_layout()
    plt.savefig('./images/correlation_heatmap.png')


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    y = df['Churn']

    X = pd.DataFrame()

    # gender encoded column
    gender_lst = []
    gender_groups = df.groupby('Gender').mean()['Churn']

    for val in df['Gender']:
        gender_lst.append(gender_groups.loc[val])

    df['Gender_Churn'] = gender_lst
    # education encoded column
    edu_lst = []
    edu_groups = df.groupby('Education_Level').mean()['Churn']

    for val in df['Education_Level']:
        edu_lst.append(edu_groups.loc[val])

    df['Education_Level_Churn'] = edu_lst

    # marital encoded column
    marital_lst = []
    marital_groups = df.groupby('Marital_Status').mean()['Churn']

    for val in df['Marital_Status']:
        marital_lst.append(marital_groups.loc[val])

    df['Marital_Status_Churn'] = marital_lst

    # income encoded column
    income_lst = []
    income_groups = df.groupby('Income_Category').mean()['Churn']

    for val in df['Income_Category']:
        income_lst.append(income_groups.loc[val])

    df['Income_Category_Churn'] = income_lst

    # card encoded column
    card_lst = []
    card_groups = df.groupby('Card_Category').mean()['Churn']

    for val in df['Card_Category']:
        card_lst.append(card_groups.loc[val])

    df['Card_Category_Churn'] = card_lst

    #X[keep_cols] = df[keep_cols]


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
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    y = df['Churn']
    X = pd.DataFrame()
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']
    X[keep_cols] = df[keep_cols]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


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
