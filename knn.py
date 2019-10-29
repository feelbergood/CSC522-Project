from sklearn.neighbors import KNeighborsClassifier


def get_models_with_ks():
    ns = {}
    for i in range(1, 100):
        n = KNeighborsClassifier(n_neighbors=i)
        ns[i] = n
    return ns


def get_model():
    neigh = KNeighborsClassifier(n_neighbors=28)
    return neigh


def get_name():
    return "KNN"
