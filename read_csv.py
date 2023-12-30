import numpy as np


def readCSV(path, skip_header=True, delimiter=',', split=False):
    data = np.loadtxt(path, dtype=str, delimiter=delimiter, skiprows=1 if skip_header else 0)
    if split:
        return data[:, :-1], data[:, -1]
    else:
        return data


def main():
    data = readCSV('./dataset/archive/DATA.csv')
    print(data)


if __name__ == '__main__':
    main()
