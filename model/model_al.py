import numpy as np


def lg(train_data, test_data, test_production):
    y_train = train_data[1]

    pred_y = np.ones(y_train.shape[0])

    production = np.ones(test_production.shape[0])

    return pred_y, production
