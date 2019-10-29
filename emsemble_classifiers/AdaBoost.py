from sklearn.ensemble import AdaBoostClassifier


def get_model():
    adaBoost = AdaBoostClassifier()
    return adaBoost


def get_name():
    return "AdaBoost"
