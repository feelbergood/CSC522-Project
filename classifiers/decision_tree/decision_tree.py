from sklearn import tree


def get_model():
    dt = tree.DecisionTreeClassifier()
    return dt


def get_name():
    return "DT"
