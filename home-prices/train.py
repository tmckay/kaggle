import csv

import numpy as np
from sklearn import datasets, linear_model


def main():
    # Read data
    with open('train.csv') as fh:
        #labels_reader = csv.reader(fh)
        labels_reader = csv.DictReader(fh)

        X_feats = []
        Y_labels = []

        for row in labels_reader:
            X_feats.append([int(row['YearBuilt']), int(row['LotArea']), int(row['1stFlrSF']), int(row['2ndFlrSF']), int(row['YearRemodAdd'])])
            Y_labels.append(row['SalePrice'])

        regr = linear_model.LinearRegression()
        regr.fit(np.array(X_feats), np.array(Y_labels).reshape(-1, 1))

        pred_year = [[1995, 7500, 800, 800, 2006]]
        pred_sale_price = int(regr.predict(pred_year)[0][0])
        print(f'For a home built in {pred_year[0]}, we predict a price of ${pred_sale_price:,}')


if __name__ == '__main__':
    main()
