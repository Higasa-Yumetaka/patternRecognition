from loadDataSet import loadDataSet


class CART:
    def __init__(self, dataset, features):
        self.dataset = dataset
        self.features = features
        self.tree = None

    @staticmethod
    def calcGini(dataset):
        # 求总样本数
        num_of_examples = len(dataset)
        labelCnt = {}
        # 遍历整个样本集合
        for example in dataset:
            # 当前样本的标签值是该列表的最后一个元素
            currentLabel = example[-1]
            # 统计每个标签各出现了几次
            if currentLabel not in labelCnt.keys():
                labelCnt[currentLabel] = 0
            labelCnt[currentLabel] += 1
        # 得到了当前集合中每个标签的样本个数后，计算它们的p值
        for key in labelCnt:
            labelCnt[key] /= num_of_examples
            labelCnt[key] = labelCnt[key] * labelCnt[key]
        # 计算Gini系数
        Gini = 1 - sum(labelCnt.values())
        return Gini

    # 提取子集合
    # 功能：从dataSet中先找到所有第axis个标签值 = value的样本
    # 然后将这些样本删去第axis个标签值，再全部提取出来成为一个新的样本集
    @staticmethod
    def create_sub_dataset(dataset, index, value):
        sub_dataset = []
        for example in dataset:
            if example[index] == value:
                current_list = example[:index]
                current_list.extend(example[index + 1:])
                sub_dataset.append(current_list)
        return sub_dataset

    # 分离
    @staticmethod
    def split_dataset(dataset, index, value):
        sub_dataset1 = []
        sub_dataset2 = []
        for example in dataset:
            if example[index] == value:
                current_list = example[:index]
                current_list.extend(example[index + 1:])
                sub_dataset1.append(current_list)
            else:
                current_list = example[:index]
                current_list.extend(example[index + 1:])
                sub_dataset2.append(current_list)
        return sub_dataset1, sub_dataset2

    def choose_best_feature(self, dataset):
        # 特征总数
        numFeatures = len(dataset[0]) - 1
        # 当只有一个特征时
        if numFeatures == 1:
            return 0, dataset[0][0]
        # 初始化最佳基尼系数
        bestGini = 1
        # 初始化最优特征
        index_of_best_feature = -1
        # 初始化最优切分点
        best_split_point = None
        # 遍历所有特征，寻找最优特征和该特征下的最优切分点
        for i in range(numFeatures):
            # 去重，每个属性值唯一
            uniqueVals = set(example[i] for example in dataset)
            # Gini字典中的每个值代表以该值对应的键作为切分点对当前集合进行划分后的Gini系数
            Gini = {}
            # 对于当前特征的每个取值
            for value in uniqueVals:
                # 先求由该值进行划分得到的两个子集
                sub_dataset1, sub_dataset2 = self.split_dataset(dataset, i, value)
                # 求两个子集占原集合的比例系数prob1 prob2
                prob1 = len(sub_dataset1) / float(len(dataset))
                prob2 = len(sub_dataset2) / float(len(dataset))
                # 计算子集1的Gini系数
                Gini_of_sub_dataset1 = self.calcGini(sub_dataset1)
                # 计算子集2的Gini系数
                Gini_of_sub_dataset2 = self.calcGini(sub_dataset2)
                # 计算由当前最优切分点划分后的最终Gini系数
                Gini[value] = prob1 * Gini_of_sub_dataset1 + prob2 * Gini_of_sub_dataset2
                # 更新最优特征和最优切分点
                if Gini[value] < bestGini:
                    bestGini = Gini[value]
                    index_of_best_feature = i
                    best_split_point = value
        return index_of_best_feature, best_split_point

    @staticmethod
    def find_label(classList):
        # 初始化统计各标签次数的字典
        # 键为各标签，对应的值为标签出现的次数
        labelCnt = {}
        for key in classList:
            if key not in labelCnt.keys():
                labelCnt[key] = 0
            labelCnt[key] += 1
        sorted_labelCnt = sorted(labelCnt.items(), key=lambda a: a[1], reverse=True)
        return sorted_labelCnt[0][0]

    def create_decision_tree(self, dataset, features, depth=0):
        # logger.info(f"深度:{depth}")
        # logger.info(f"当前特征集:{features}")
        # 求出训练集所有样本的标签
        label_list = [example[-1] for example in dataset]
        # 先写两个递归结束的情况：
        # 若当前集合的所有样本标签相等（即样本已被分“纯”）
        # 则直接返回该标签值作为一个叶子节点
        # logger.info(f"label_list:{label_list}")
        if label_list.count(label_list[0]) == len(label_list):
            # logger.info(f"叶子节点:{label_list[0]}")
            return label_list[0]
        # 若训练集的所有特征都被使用完毕，当前无可用特征，但样本仍未被分“纯”
        # 则返回所含样本最多的标签作为结果
        if len(dataset[0]) == 1:
            # logger.info(f"叶子节点:{self.find_label(label_list)}")
            return self.find_label(label_list)
        # 下面是正式建树的过程
        # 选取进行分支的最佳特征的下标和最佳切分点
        index_of_best_feature, best_split_point = self.choose_best_feature(dataset)
        # 得到最佳特征
        best_feature = features[index_of_best_feature]
        # 初始化决策树
        decision_tree = {best_feature: {}}
        # 使用过当前最佳特征后将其删去
        features = features[:index_of_best_feature] + features[index_of_best_feature + 1:]
        # 子特征 = 当前特征
        # sub_features = features.copy()
        # 递归调用create_decision_tree去生成新节点
        # 生成由最优切分点划分出来的二分子集
        sub_dataset1, sub_dataset2 = self.split_dataset(dataset, index_of_best_feature, best_split_point)
        # 如果划分出来的子集为空，则直接返回该子集中所含样本最多的标签作为叶子节点
        if len(sub_dataset2) == 0 or len(sub_dataset1) == 0:
            return self.find_label(label_list)
        # 构造左子树
        # logger.info(f"左子树:{best_feature} = {best_split_point}, {len(sub_dataset1)}")
        decision_tree[best_feature][best_split_point] = self.create_decision_tree(sub_dataset1, features, depth + 1)
        # 构造右子树
        # logger.info(f"右子树:{best_feature} != {best_split_point}, {len(sub_dataset2)}")
        decision_tree[best_feature]['others'] = self.create_decision_tree(sub_dataset2, features, depth + 1)
        self.tree = decision_tree
        return decision_tree

    # 用上面训练好的决策树对新样本分类
    def classify(self, decision_tree, features, test_example):
        # 根节点代表的属性
        classLabel = None
        first_feature = list(decision_tree.keys())[0]
        # second_dict是第一个分类属性的值（也是字典）
        second_dict = decision_tree[first_feature]
        # 树根代表的属性，所在属性标签中的位置，即第几个属性
        index_of_first_feature = features.index(first_feature)
        # 对于second_dict中的每一个key
        for key in second_dict.keys():
            # 不等于others的key
            if key != 'others':
                if test_example[index_of_first_feature] == key:
                    # 若当前second_dict的key的value是一个字典
                    if type(second_dict[key]).__name__ == 'dict':
                        # 则需要递归查询
                        classLabel = self.classify(second_dict[key], features, test_example)
                    # 若当前second_dict的key的value是一个单独的值
                    else:
                        # 则就是要找的标签值
                        classLabel = second_dict[key]
                # 如果测试样本在当前特征的取值不等于key，就说明它在当前特征的取值属于others
                else:
                    # 如果second_dict['others']的值是个字符串，则直接输出
                    if isinstance(second_dict['others'], str):
                        classLabel = second_dict['others']
                    # 如果second_dict['others']的值是个字典，则递归查询
                    else:
                        classLabel = self.classify(second_dict['others'], features, test_example)
        return classLabel

    def predict(self, test_data):
        # 对测试集中的每一个测试样本进行预测
        classLabelAll = []
        for test_example in test_data:
            classLabelAll.append(self.classify(self.tree, self.features, test_example))
        return classLabelAll

    def fit(self):
        self.tree = self.create_decision_tree(self.dataset, self.features)
        return self.tree

    def __repr__(self):
        return "CART"


def main():
    path = '../dataset/archive/DATA.csv'
    train_data, test_data, Feature = loadDataSet(path, separate=False)
    clf = CART(train_data, Feature)
    clf.fit()
    predict = clf.predict(test_data)
    correct = 0
    for i in range(len(test_data)):
        if predict[i] == test_data[i][-1]:
            correct += 1
    print("{} accuracy: {:.2f}".format(clf.__repr__(), correct * 100 / len(test_data)))


if __name__ == "__main__":
    main()
