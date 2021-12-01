'''
Module to store constants required in churn_library.py file

Author: Xavier 
Date: Nov 2021
'''


data_pth = './data/bank_data.csv'

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

eda_imgs = {
    'images/eda/churn_distribution.png',
    'images/eda/marital_status_distribution.png',
    'images/eda/customer_age_distribution.png',
    'images/eda/heatmap.png',
    'images/eda/total_transaction_distribution.png'
}

results_imgs = {
    'images/results/rf_classification_report.png',
    'images/results/lr_classification_report.png',
    'images/results/feature_importances.png',
    'images/results/lr_roc_curve.png',
    'images/results/rf_roc_curve.png'
}

