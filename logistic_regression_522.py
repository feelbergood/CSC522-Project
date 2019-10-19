from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults

def get_model(x, y):
    logisticRegr = LogisticRegression()
    model = logisticRegr.fit(x_train, y_train)
    return model
# predictions = logisticRegr.predict(x_test)

