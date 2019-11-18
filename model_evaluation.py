import pandas as pd
import logistic_regression
import knn
import svm
import baseline
from svm_classifiers import svm_c, svm_linear, svm_nu
from decision_tree import decision_tree
from emsemble_classifiers import AdaBoost, Bagging, RandomForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import confusion_matrix


models = [baseline, logistic_regression, knn, decision_tree, svm, svm_c, svm_linear, svm_nu, AdaBoost, Bagging, RandomForest]
# knn_models = knn.get_models_with_ks()
# knn_accuracies = {}


def make_predictions(model, x, y):
    y_pred = cross_val_predict(model, x, y, cv=10)
    return y_pred


def evaluate_predictions(model, name, x, y):
    y_pred = make_predictions(model, x, y)
    print("confusion matrix: \n", confusion_matrix(y, y_pred))

    def tn(y, y_pred): return confusion_matrix(y, y_pred)[0, 0]

    def fp(y, y_pred): return confusion_matrix(y, y_pred)[0, 1]

    def fn(y, y_pred): return confusion_matrix(y, y_pred)[1, 0]

    def tp(y, y_pred): return confusion_matrix(y, y_pred)[1, 1]

    tp, tn, fp, fn = int(tn(y, y_pred)), int(tp(y, y_pred)), int(fn(y, y_pred)), int(fp(y, y_pred))
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision) * (recall) / (precision + recall)
    # knn_accuracies[name] = accuracy
    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)
    evaluation_metrics = pd.read_csv("model_evaluation_results/train_evaluation.csv", index_col=0)
    model_name = name
    new_row = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": accuracy, "precision": precision, "recall": recall,
               "f1": f1}
    evaluation_metrics.loc[model_name] = new_row
    evaluation_metrics.to_csv("model_evaluation_results/train_evaluation.csv")
    return name, accuracy, precision, recall, f1


def get_xy():
    data = pd.read_csv('output/team_seasons_classified_1.csv')
    x = data[['o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_oreb',
              'o_dreb', 'o_reb', 'o_asts', 'o_pf', 'o_stl', 'o_to', 'o_blk', 'o_pts', 'd_fgm', 'd_fga', 'd_ftm',
              'd_fta', 'd_oreb',
              'd_dreb', 'd_reb', 'd_asts', 'd_pf', 'd_stl', 'd_to', 'd_blk', 'd_pts', 'pace']]
    y = data['class']
    mapper = DataFrameMapper([(x.columns, StandardScaler())])
    x = mapper.fit_transform(x, 4)
    mapper = DataFrameMapper([(y, LabelEncoder())])
    y = mapper.fit_transform(y, 4).ravel()
    return x, y


def run_me():
    x, y = get_xy()
    for model in models:
        print("--------------" + model.get_name() + "--------------")
        evaluate_predictions(model.get_model(), model.get_name(), x, y)


run_me()