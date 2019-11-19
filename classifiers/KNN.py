from sklearn.neighbors import KNeighborsClassifier
from classifiers.Model import Model


class KNNModel(Model):
    def __init__(self):
        self.name = "KNN"
        self.model = KNeighborsClassifier(n_neighbors=28)


class TunedKNNModel(Model):
    def __init__(self):
        self.name = "KNN"
        self.model = KNeighborsClassifier(n_neighbors=28)

