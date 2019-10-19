import pandas as pd 
import logistic_regression_522 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn_pandas import DataFrameMapper



def build_model():
    return logistic_regression_522.get_model()

def make_predictions(x, y):
    model = build_model()
    y_pred = cross_val_predict(model, x, y, cv=10)
    return y_pred 

def evaluate_predictions(x, y):
    y_pred = make_predictions(x, y)
    def tn(y, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
    def fp(y, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
    def fn(y, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
    def tp(y, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
    tp, tn, fp, fn = make_scorer(tp),make_scorer(tn), make_scorer(fp), make_scorer(fn)
    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
               'fp': make_scorer(fp), 'fn': make_scorer(fn)}
    accuracy = tp + tn,
    precision = tp / (tp + fp),
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(scoring)
    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)    
    print("f1: ", f1)


def run_me():
    data = pd.read_csv('output/team_seasons_classified_1_train.csv')
    x = data[[ 'o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_oreb',
       'o_dreb', 'o_reb', 'o_asts', 'o_pf', 'o_stl', 'o_to', 'o_blk', 'o_pts', 'd_fgm', 'd_fga', 'd_ftm', 'd_fta', 'd_oreb',
       'd_dreb', 'd_reb', 'd_asts', 'd_pf', 'd_stl', 'd_to', 'd_blk',  'd_pts', 'pace']]
    y = data.class
    mapper = DataFrameMapper([(df.columns, StandardScaler())])
    scaled_features = mapper.fit_transform(df.copy(), 4)
    scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

    evaluate_predictions(x,y)


run_me()