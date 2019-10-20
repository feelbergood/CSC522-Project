from sklearn import tree


def get_model():
    decision_tree = tree.DecisionTreeClassifier()
    return decision_tree