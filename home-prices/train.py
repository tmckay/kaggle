import csv


def main():
    # Read data
    with open('train.csv') as fh:
        #labels_reader = csv.reader(fh)
        labels_reader = csv.DictReader(fh)

        for row in labels_reader:
            print(row.keys())
            numeric = []
            for k,v in row.items():
                if v.isnumeric():
                    numeric.append(k)
            print('Numeric')
            print(numeric)
            break

if __name__ == '__main__':
    main()
