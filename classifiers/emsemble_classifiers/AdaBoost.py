from sklearn.ensemble import AdaBoostClassifier
from classifiers.Model import Model


class AdaBoostModel(Model):
    def __init__(self):
        self.name = "AdaBoost"
        self.model = AdaBoostClassifier()
