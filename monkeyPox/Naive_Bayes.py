from collections import defaultdict

import numpy as np
from loguru import logger
from loadDataSet import loadDataSet

T = 20  # 迭代次数


class NaiveBayes:
    def __init__(self, X, y):
        self._prior_prob = defaultdict(float)  # 先验概率
        self._likelihood = defaultdict(defaultdict)  # 条件概率
        self._ck_counter = defaultdict(float)  # 类别计数
        self._Sj = defaultdict(float)
        self._X = X
        self._y = y

    def fit(self):
        n_samples, n_features = self._X.shape
        ck, ck_cnt = np.unique(self._y, return_counts=True)
        self._ck_counter = dict(zip(ck, ck_cnt))
        for label, num_label in self._ck_counter.items():
            self._prior_prob[label] = (num_label + 1) / (n_samples + ck.shape[0])
        ck_idx = []
        for label in ck:
            label_idx = np.squeeze(np.argwhere(self._y == label))
            ck_idx.append(label_idx)
        ck_idx = [np.atleast_1d(arr) for arr in ck_idx]
        # 遍历每个类别
        for label, idx in zip(ck, ck_idx):
            xdata = self._X[idx]
            # 记录该类别所有特征对应的概率
            label_likelihood = defaultdict(defaultdict)
            for i in range(n_features):
                feature_val_prob = defaultdict(float)
                # 获取该列特征可能的取值和每个取值出现的次数
                feature_val, feature_cnt = np.unique(xdata[:, i],
                                                     return_counts=True)
                self._Sj[i] = feature_val.shape[0]
                feature_counter = dict(zip(feature_val, feature_cnt))
                for fea_val, cnt in feature_counter.items():
                    # 计算该列特征每个取值的概率，做了拉普拉斯平滑，即为了计算P（x | y）
                    feature_val_prob[fea_val] = (cnt + 1) / (self._ck_counter[label] + self._Sj[i])
                label_likelihood[i] = feature_val_prob
            self._likelihood[label] = label_likelihood

    def classify(self, x):
        # 保存分类到每个类别的后验概率，即计算P（y|x）
        post_prob = defaultdict(float)
        # 遍历每个类别计算后验概率
        for label, label_likelihood in self._likelihood.items():
            prob = np.log(self._prior_prob[label])
            # 遍历样本每一维特征
            for i, fea_val in enumerate(x):
                feature_val_prob = label_likelihood[i]
                # 如果该特征值出现在训练集中则直接获取概率
                if fea_val in feature_val_prob:
                    prob += np.log(feature_val_prob[fea_val])
                else:
                    # 如果该特征没有出现在训练集中则采用拉普拉斯平滑计概率
                    laplace_prob = 1 / (self._ck_counter[label] + self._Sj[i])
                    prob += np.log(laplace_prob)
            post_prob[label] = prob
        prob_list = list(post_prob.items())
        prob_list.sort(key=lambda v: v[1], reverse=True)
        # 返回后验概率最大的类别作为预测类别
        return prob_list[0][0]

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.classify(x))
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)

    def __repr__(self):
        return "NaiveBayes"


def main():
    xtrain, xtest, ytrain, ytest, feature = loadDataSet("../dataset/archive/DATA.csv", separate=True)
    clf = NaiveBayes(xtrain, ytrain)
    clf.fit()
    logger.info("{} Accuracy: {:.2f}%".format(clf.__repr__(), clf.score(xtest, ytest) * 100))


if __name__ == "__main__":
    main()
