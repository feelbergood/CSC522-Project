import pandas as pd 
import logistic_regression_522 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import cross_val_score


data = pd.read_csv('output/team_seasons_classified_1_train.csv')

def build_model():
    return logistic_regression_522.get_model()

def make_predictions(x, y):
    model = build_model()
    y_pred = cross_val_predict(clf, x, y, cv=10)
    return y_pred 

def evaluate_predictions(x, y):
    y_pred = make_predictions(x, y)
    def tn(y, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
    def fp(y, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
    def fn(y, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
    def tp(y, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
               'fp': make_scorer(fp), 'fn': make_scorer(fn)}

    print(scoring)
    
def run_me():
    evaluate_predictions(x,y)