from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression


def rf(train_data, test_data):
    x_train = train_data[0]
    y_train = train_data[1]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x_train, y_train)

    # 用得到的模型进行结果预测
    predicte = rfr.predict(test_data)

    return predicte


def lg(train_data, test_data, test_prodution):
    x_train = train_data[0]
    y_train = train_data[1]

    model = LogisticRegression()
    model.fit(x_train, y_train)

    pred_Y = model.predict(test_data)

    prodution = model.predict(test_prodution)

    return pred_Y, prodution
