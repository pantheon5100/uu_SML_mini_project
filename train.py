import csv

import numpy as np
from sklearn.model_selection import KFold

from utils.utils import load_data
from model.model_random import rf, lg


class Train:
    def __init__(self, model):
        super(Train, self).__init__()
        self.model = model

        # Load data
        loaded_data = load_data()
        self.tr_data = loaded_data[0][0]
        self.tr_label = loaded_data[0][1]
        self.te_data = loaded_data[1]

    def start(self, k=10):
        # create result recorder
        store_file = open('Result.csv', 'w', newline='')
        writer = csv.DictWriter(store_file, ['Fold', 'Acc_tr', "In production"])
        writer.writeheader()

        kf = KFold(10)
        kf_data = kf.split(self.tr_data)
        for K, [train_index, test_index] in enumerate(kf_data):
            X_train = self.tr_data[train_index]
            y_train = self.tr_label[train_index]
            X_test = self.tr_data[test_index]
            y_test = self.tr_label[test_index]

            # Construct our model by instantiating the class defined above
            pred, prodution = self.model([X_train, y_train], X_test, self.te_data)
            # pred process ...
            pred_string = ""
            for p in prodution.astype(np.int):
                pred_string += str(p)

            acc = np.sum(y_test==pred)/y_test.shape[0]
            writer.writerow({"Fold": K,
                             "Acc_tr": acc,
                             "In production": pred_string})

        store_file.close()


train = Train(lg)
train.start()
