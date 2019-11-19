from sklearn.linear_model import LogisticRegression
from classifiers.Model import Model


class LRModel(Model):
    def __init__(self):
        self.name = "LR"
        self.model = LogisticRegression(solver='lbfgs', max_iter=2000)
