from sklearn.neighbors import KNeighborsClassifier

def get_model():
    neigh = KNeighborsClassifier(n_neighbors=3)
    # model = logisticRegr.fit()
    return neigh
