from classifiers.Model import Model
from sklearn.neural_network import MLPClassifier


class MLPModel(Model):
    def __init__(self):
        self.name = "MLP"
        self.model = MLPClassifier()
