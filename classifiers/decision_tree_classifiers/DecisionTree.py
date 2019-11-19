from sklearn import tree
from classifiers.Model import Model


class DecisionTreeModel(Model):
    def __init__(self):
        self.name = "DT"
        self.model = tree.DecisionTreeClassifier()
