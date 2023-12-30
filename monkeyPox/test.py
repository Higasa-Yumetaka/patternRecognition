import numpy as np
from loguru import logger
from loadDataSet import loadDataSet
from CART import CART


class AdaBoost:
    def __init__(self, dataset, feature, T):
        self.dataset = dataset  # 数据集 dataset
        self.feature = feature  # 特征 feature
        self.classifier = []  # 弱分类器 weak classifier
        self.Weight_classifier = []  # 弱分类器的系数 weak classifier coefficient
        self.error = []  # 弱分类器的误差 weak classifier error
        self.T = T  # 弱分类器的个数 number of weak classifiers
        self.Weight_sample = [1 / len(dataset) for _ in range(len(dataset))]  # 样本权重 sample weight

    def selectSample(self):
        select_index = np.random.choice(len(self.dataset), int(len(self.dataset) / self.T), p=self.Weight_sample)
        select_samples = []
        for i in select_index:
            select_samples.append(self.dataset[i])
        return select_samples

    def fit(self):
        for i in range(self.T):
            data = self.selectSample()
            cart = CART(data, self.feature)
            cart.fit()

            predict = cart.predict(self.dataset)
            error = 0
            for j in range(len(predict)):
                if predict[j] != self.dataset[j][-1]:
                    error += self.Weight_sample[j]
            # 分类器效果差于随机分类器
            if error > 0.5:
                continue
            else:
                self.classifier.append(cart)
                # 计算分类器系数
                weight_classifier = 0.5 * np.log((1 - error) / error)
                self.Weight_classifier.append(weight_classifier)
                # 计算样本权重
                for j in range(len(predict)):
                    if predict[j] == self.dataset[j][-1]:
                        self.Weight_sample[j] = self.Weight_sample[j] * np.exp(-weight_classifier)
                    else:
                        self.Weight_sample[j] = self.Weight_sample[j] * np.exp(weight_classifier)
                # 样本权重归一化
                self.Weight_sample = self.Weight_sample / np.sum(self.Weight_sample)
                self.error.append(error)
                logger.info("第{}个弱分类器训练完成，正确率{:.2f}".format(i + 1, 1 - error))

    def classify(self, test_data):
        predict_list = []
        for sample in test_data:
            predict = []  # 每个弱分类器的预测结果
            weight = []  # 每个预测结果的权重
            for i in range(len(self.classifier)):
                pred = self.classifier[i].classify(self.classifier[i].tree, self.feature, sample)
                if pred in predict:
                    index = predict.index(pred)
                    weight[index] += self.Weight_classifier[i]
                else:
                    predict.append(pred)
                    weight.append(self.Weight_classifier[i])
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


def main():
    path = '../dataset/archive/DATA.csv'
    train_data, test_data, Feature = loadDataSet(path)
    adaboost = AdaBoost(train_data, Feature, 100)
    adaboost.fit()
    error = 0
    for i in range(len(adaboost.error)):
        error += adaboost.error[i]
    error = error / len(adaboost.error)
    print("基分类器平均正确率: {:.2f}".format(1 - error))
    print("AdaBoost分类器正确率: {:.2f}".format(adaboost.score(test_data)))


if __name__ == "__main__":
    main()
