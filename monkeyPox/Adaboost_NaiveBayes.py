import numpy as np
from loguru import logger

from Naive_Bayes import NaiveBayes as NB
from loadDataSet import loadDataSet


class AdaBoost:
    def __init__(self, T, dataset):
        self._dataset = dataset  # 数据集 dataset
        self._classifier = []  # 弱分类器 weak classifier
        self._Weight_classifier = []  # 弱分类器的系数 weak classifier coefficient
        self._error = []  # 弱分类器的误差 weak classifier error
        self._T = T  # 弱分类器的个数 number of weak classifiers
        self._Weight_sample = [1 / len(dataset) for _ in range(len(dataset))]  # 样本权重 sample weight
        self._X = np.array(self._dataset)[:, :-1]
        self._y = np.array(self._dataset)[:, -1]

    def selectSample(self):
        select_index = np.random.choice(len(self._dataset), int(len(self._dataset) / 2), p=self._Weight_sample)
        select_samples = []
        for i in select_index:
            select_samples.append(self._dataset[i])
        return select_samples

    def fit(self):
        for i in range(self._T):
            data = self.selectSample()
            data = np.array(data)
            Xtrain = data[:, :-1]
            ytrain = data[:, -1]
            clf = NB(Xtrain, ytrain)
            clf.fit()
            predict = clf.predict(self._dataset)
            error = 0
            for j in range(len(predict)):
                if predict[j] != self._dataset[j][-1]:
                    error += self._Weight_sample[j]
            # 分类器效果差于随机分类器
            if error > 0.5:
                continue
            else:
                self._classifier.append(clf)
                # 计算分类器系数
                weight_classifier = 0.5 * np.log((1 - error) / error)
                self._Weight_classifier.append(weight_classifier)
                # 计算样本权重
                for j in range(len(predict)):
                    if predict[j] == self._dataset[j][-1]:
                        self._Weight_sample[j] = self._Weight_sample[j] * np.exp(-weight_classifier)
                    else:
                        self._Weight_sample[j] = self._Weight_sample[j] * np.exp(weight_classifier)
                # 样本权重归一化
                self._Weight_sample = self._Weight_sample / np.sum(self._Weight_sample)
                self._error.append(error)
                logger.info("第{}个弱分类器训练完成，正确率{:.2f}".format(i + 1, 1 - error))

    def classify(self, test_data):
        predict_list = []
        for sample in test_data:
            predict = []  # 每个弱分类器的预测结果
            weight = []  # 每个预测结果的权重
            for i in range(len(self._classifier)):
                pred = self._classifier[i].classify(sample)
                if pred in predict:
                    index = predict.index(pred)
                    weight[index] += self._Weight_classifier[i]
                else:
                    predict.append(pred)
                    weight.append(self._Weight_classifier[i])
            index = weight.index(max(weight))
            predict_list.append(predict[index])
        return predict_list

    def score(self, test_data):
        predict = self.classify(test_data)
        correct = 0
        for i in range(len(test_data)):
            if predict[i] == test_data[i][-1]:
                correct += 1
        return correct / len(test_data)

    def base_score(self):
        return 1 - np.average(self._error)

    def __repr__(self):
        return "AdaBoost"

    def __method__(self):
        return "NaiveBayes"


def main():
    path = '../dataset/archive/DATA.csv'
    train_data, test_data, Feature = loadDataSet(path)
    adaboost = AdaBoost(20, train_data)
    adaboost.fit()
    logger.info("{} 基分类器平均正确率: {:.2f}%".format(adaboost.__method__(), adaboost.base_score() * 100))
    logger.info("{}_{} 分类器正确率: {:.2f}%".format(adaboost.__repr__(), adaboost.__method__(),
                                                     adaboost.score(test_data) * 100))


if __name__ == "__main__":
    main()
