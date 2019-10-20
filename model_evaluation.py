import pandas as pd 
import logistic_regression_522 
import knn_522
import decision_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import confusion_matrix


def build_model():
    # return logistic_regression_522.get_model()
    return knn_522.get_model()


def make_predictions(x, y):
    model = build_model()
    y_pred = cross_val_predict(model, x, y, cv=10)
    return y_pred


def evaluate_predictions(x, y):
    y_pred = make_predictions(x, y)
    print(confusion_matrix(y, y_pred))
    def tn(y, y_pred): return confusion_matrix(y, y_pred)[0, 0]
    def fp(y, y_pred): return confusion_matrix(y, y_pred)[0, 1]
    def fn(y, y_pred): return confusion_matrix(y, y_pred)[1, 0]
    def tp(y, y_pred): return confusion_matrix(y, y_pred)[1, 1]
    tp, tn, fp, fn = int(tn(y, y_pred)), int(tp(y, y_pred)), int(fn(y, y_pred)),int(fp(y, y_pred))
    accuracy = (tp + tn)/(tp+ tn+ fp+ fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision) * (recall) / (precision + recall)
    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)    
    print("f1: ", f1)
    evaluation_metrics = pd.read_csv("model_evaluation_metrics/train_evaluation.csv", index_col = 0)
    model_name = 'undefined'
    new_row = {"tp":tp, "tn":tn, "fp": fp, "fn": fn, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    evaluation_metrics.loc[model_name] = new_row
    evaluation_metrics.to_csv("model_evaluation_metrics/train_evaluation.csv")

# def evaluate_model(model, x, y):


def run_me():
    data = pd.read_csv('output/team_seasons_classified_1_train.csv')
    
    x = data[[ 'o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_oreb',
       'o_dreb', 'o_reb', 'o_asts', 'o_pf', 'o_stl', 'o_to', 'o_blk', 'o_pts', 'd_fgm', 'd_fga', 'd_ftm', 'd_fta', 'd_oreb',
       'd_dreb', 'd_reb', 'd_asts', 'd_pf', 'd_stl', 'd_to', 'd_blk',  'd_pts', 'pace']]
    y = data['class']
    mapper = DataFrameMapper([(x.columns, StandardScaler())])
    x = mapper.fit_transform(x, 4)
    mapper = DataFrameMapper([(y, LabelEncoder())])
    y = mapper.fit_transform(y, 4).ravel()
    evaluate_predictions(x,y)


run_me()