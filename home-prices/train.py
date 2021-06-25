import csv


def main():
    # Read data
    with open('train.csv') as fh:
        #labels_reader = csv.reader(fh)
        labels_reader = csv.DictReader(fh)

        for row in labels_reader:
            keys = row.keys()
            print(f'Number of potential features: {len(keys)}')
            print(keys)
            numeric = []
            for k,v in row.items():
                if v.isnumeric():
                    numeric.append(k)
            print(f'Number of numeric features: {len(numeric)}')
            print(numeric)
            break

if __name__ == '__main__':
    main()
