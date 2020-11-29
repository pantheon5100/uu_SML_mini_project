import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


def read_data(train_path, test_path):
    # read train and test data from local path
    # return the form need for models
    train = pd.read_csv(train_path)
    X_train = train.iloc[:, :-1]  # input
    y_train = train.iloc[:, -1]  # output
    X_test = pd.read_csv(test_path)
    return X_train, y_train, X_test


def gscv(X_train, y_train):
    # using Grid Search Cross Validation to tune hyper-parameters of kNN model on training data
    # return the tuned best model
    parameters = {'n_neighbors': np.arange(1, 50 + 1),
                  'weights': ('uniform', 'distance'),
                  'p': [1, 2]}
    model_kNN = KNeighborsClassifier()
    CV = KFold(n_splits=10, shuffle=True, random_state=0)

    gscv = GridSearchCV(model_kNN, parameters, cv=CV)
    gscv.fit(X_train, y_train)
    return gscv.best_estimator_


def KFoldCV(X_train, y_train, K=10, random_seed=0):
    cv = KFold(n_splits=K, shuffle=True, random_state=random_seed)
    #   acc is mean accuracy of 10 fold cross validation
    acc = 0
    model_kNN_gscvBest = gscv(X_train, y_train)

    for train_index, val_index in cv.split(X_train):
        x_train, x_val = X_train.iloc[train_index], X_train.iloc[val_index]
        Y_train, Y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model_kNN_gscvBest.fit(x_train, Y_train)
        acc += model_kNN_gscvBest.score(x_val, Y_val)
    acc /= K
    return acc


def kNN(X_train, y_train, X_test):
    # tune k Nearest Neighbors model parameters with gscv
    # return prediction on test data
    model_kNN_gscvBest = gscv(X_train, y_train)
    model_kNN_gscvBest.fit(X_train, y_train)
    prediction = model_kNN_gscvBest.predict(X_test)
    return prediction


if __name__ == "__main__":
    X_train, y_train, X_test = read_data("training_data.csv", "songs_to_classify.csv")
    acc = KFoldCV(X_train, y_train, K=10, random_seed=0)
    print("10 fold cross validation mean accuracy on training data: ", acc)
    prediction = kNN(X_train, y_train, X_test)
    print("Prediction on test data:")
    print(prediction)