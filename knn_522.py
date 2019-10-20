from sklearn.neighbors import KNeighborsClassifier


def get_model():
    neigh = KNeighborsClassifier(n_neighbors=5)
    return neigh


def get_name():
    return "KNN"
