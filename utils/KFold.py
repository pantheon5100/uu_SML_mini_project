import random
import numpy as np


class KFold:
    """
    K-fold cross validation

    K is a integer.
    Data is the label of the data, such as [1, 0, 0, ...].
    """
    def __init__(self, K, Data):
        super(KFold, self).__init__()
        self.K = K
        self.Data_size = Data.shape[0]

    def get_data_iter(self):
        size_per_iter = int(self.Data_size/self.K)
        remainder = self.Data_size%self.K
        sample = [i for i in range(1,self.Data_size)]
        cp_sample = sample.copy()

        train_ind = []
        test_ind = []
        for i in range(self.K):
            selection = random.sample(sample, size_per_iter)
            test_ind.append(selection)
            del(cp_sample[selection])
            tmp_sample = sample.copy()
            del(tmp_sample[selection])
            train_ind.append(tmp_sample)
            # 训练集,test_num个做测试集
        return train_ind, test_ind
        pass


if __name__ == "__main__":
    kfold = KFold(6, np.array([i for i in range(1, 100)]))
    train_ind, test_ind = kfold.get_data_iter()
