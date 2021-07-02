import csv
import random

import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def main():
    # Read data
    with open('train.csv') as fh:
        labels_reader = csv.DictReader(fh)

        X_feats = []
        Y_labels = []

        for row in labels_reader:
            X_feats.append([
                int(row['YearBuilt']),
                int(row['LotArea']),
                int(row['1stFlrSF']),
                int(row['2ndFlrSF']),
                int(row['YearRemodAdd']),
                int(row['OverallQual'])
            ])
            Y_labels.append(int(row['SalePrice']))

        split = int(len(X_feats) * .5)
        X_feats_train = X_feats[:split]
        X_feats_test = X_feats[split:]

        Y_labels_train = Y_labels[:split]
        Y_labels_test = Y_labels[split:]

        regr = linear_model.LinearRegression()
        regr.fit(np.array(X_feats_train), np.array(Y_labels_train).reshape(-1, 1))

        single_features = [[1995, 7500, 800, 800, 2006, 8]]
        pred_sale_price = int(regr.predict(single_features)[0][0])
        print(f'For a home with features {single_features[0]}, we predict a price of ${pred_sale_price:,}')

        test_predictions = regr.predict(X_feats_test)
        mse = mean_squared_error(test_predictions, Y_labels_test)
        print(f'Mean squared error is {mse}')

        mape = mean_absolute_percentage_error(test_predictions, Y_labels_test) * 100
        print(f'Mean absolute percentage error {mape:.2f}%')

if __name__ == '__main__':
    main()
