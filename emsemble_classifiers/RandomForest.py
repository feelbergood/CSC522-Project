from sklearn.ensemble import RandomForestClassifier


def get_model():
    rfc = RandomForestClassifier(n_estimators=100)
    return rfc


def get_name():
    return "RandomForest"
