import pandas as pd
from classifiers.Baseline import BaselineModel
from classifiers.KNN import KNNModel

baseline = BaselineModel()
knn = KNNModel()

models = [baseline, knn]
# , logistic_regression, knn, decision_tree, svm_c, svm_linear, svm_nu, AdaBoost, Bagging, RandomForest]
# knn_models = knn.get_models_with_ks()
# knn_accuracies = {}


def get_xy():
    data = pd.read_csv('preprocessed_data/team_seasons_classified_1.csv')
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return x, y


def run_me():
    x, y = get_xy()
    for model in models:
        model.get_and_save_performance(x, y)
        print("--------------" + model.get_name() + "--------------")
        print(model.get_confusion_matrix())
        print(model.get_performance())


run_me()