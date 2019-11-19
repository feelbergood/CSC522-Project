from sklearn.ensemble import RandomForestClassifier
from classifiers.Model import Model


class RandomForestModel(Model):
    def __init__(self):
        self.name = "RandomForest"
        self.model = RandomForestClassifier(n_estimators=100)
