from sklearn.ensemble import RandomForestClassifier

def get_model():
    rf = RandomForestClassifier(n_estimators=8)
    return rf


def get_name():
    return "randomForest"