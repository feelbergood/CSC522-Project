from sklearn.neighbors import KNeighborsClassifier

def get_model():
    neigh = KNeighborsClassifier(n_neighbors=5)
    # model = logisticRegr.fit()
    return neigh
