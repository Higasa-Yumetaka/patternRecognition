import csv
import numpy as np
from sklearn.model_selection import train_test_split


def loadDataSet(filename, separate=False, test_size=0.1):
    with open(filename, 'r') as file:
        Data = np.array(list(csv.reader(file, delimiter=',', quotechar='"')))
        Data = Data[:5000, 1:]
        Feature = Data[0, :-1]
        Data = Data[1:, :]
        if separate:
            xtrain, xtest, ytrain, ytest = train_test_split(Data[:, :-1], Data[:, -1], test_size=test_size,
                                                            random_state=0)
            return xtrain, xtest, ytrain, ytest, Feature
        else:
            train_data, test_data = train_test_split(Data, test_size=test_size, random_state=0)
            train_Data = train_data[1:, :]
            test_Data = test_data[1:, :]
            train_Data = train_Data.tolist()
            test_Data = test_Data.tolist()
            Feature = Feature.tolist()
            return train_Data, test_Data, Feature
