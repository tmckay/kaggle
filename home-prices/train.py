import csv


def main():
    # Read data
    with open('train.csv') as fh:
        labels_reader = csv.reader(fh)

        for row in labels_reader:
            print(row)
            break

if __name__ == '__main__':
    main()
