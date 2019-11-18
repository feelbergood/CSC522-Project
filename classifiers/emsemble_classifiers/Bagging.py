from sklearn.ensemble import BaggingClassifier


def get_model():
    bagging = BaggingClassifier()
    return bagging


def get_name():
    return "BaggingClassifier"
