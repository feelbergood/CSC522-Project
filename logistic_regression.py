from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults


def get_model():
    logisticRegr = LogisticRegression()
    # model = logisticRegr.fit()
    return logisticRegr
# predictions = logisticRegr.predict(x_test)


def get_name():
    return "LR"
