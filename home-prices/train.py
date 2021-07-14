import argparse
import csv
import random

import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    max_error
)

_CONFIG = {
    'FEATURES_FILE': 'train.csv',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-features', action='store_true')
    args = parser.parse_args()

    if args.show_features:
        show_features()
    else:
        train()


def show_features():
    """Prints all potential features names from data and the first row of data"""
    with open(_CONFIG['FEATURES_FILE']) as fh:
        labels_reader = csv.DictReader(fh)

        for row in labels_reader:
            for col, val in row.items():
                print(f'{col}: {val}')
            break  # just show one row


def pe20(preds, labels):
    """Generates a metric for the percent of predictions within 20% i.e. <=.20"""
    diff = np.abs(preds - labels)
    error = diff / preds
    return np.count_nonzero(error < .20) / np.count_nonzero(error)


def metrics(predictions, labels):
    mse = mean_squared_error(predictions, labels)
    mape = mean_absolute_percentage_error(predictions, labels) * 100
    mae = median_absolute_error(predictions, labels)
    r2 = r2_score(predictions, labels)
    max_error_amt = max_error(predictions, labels)
    pe_20 = pe20(predictions, labels)

    print(f'MSE {mse:,.1f} : MAPE {mape:.2f}% : MAE {mae:,.2f} : R2 {r2:.4f} : Max error {max_error_amt:,.1f} : PE20 {pe_20:.3f}')
    print()



def train():
    """Trains a model, makes predictions and generates metrics"""

    features = (
        'YearBuilt',
        'LotArea',
        '1stFlrSF',
        '2ndFlrSF',
        'YearRemodAdd',
        'OverallQual',
        'OverallCond',
        'TotalBsmtSF',
        'SalePrice'
    )

    # Read data
    with open(_CONFIG['FEATURES_FILE']) as fh:
        labels_reader = csv.DictReader(fh)

        all_features = []
        for row in labels_reader:
            all_features.append(
                {feat: int(row[feat]) for feat in features}
            )

    for idx in range(1, len(features)):

        X_feats = []
        Y_labels = []

        included_features = features[:idx]

        print('FEATURES: ' + ' | '.join(included_features))

        for row in all_features:
            X_feats.append(
                [int(row[feat]) for feat in included_features]
            )
            Y_labels.append(int(row['SalePrice']))

        split = int(len(X_feats) * .5)
        X_feats_train = X_feats[:split]
        X_feats_test = X_feats[split:]

        Y_labels_train = Y_labels[:split]
        Y_labels_test = Y_labels[split:]

        regr = linear_model.LinearRegression()
        regr.fit(np.array(X_feats_train), np.array(Y_labels_train).reshape(-1, 1))

        test_predictions = regr.predict(X_feats_test)

        metrics(test_predictions, Y_labels_test)

if __name__ == '__main__':
    main()
