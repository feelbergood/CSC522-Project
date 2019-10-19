import pandas as pd 
import logistic_regression_522 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import cross_val_score


data = pd.read_csv('output/team_seasons_classified_1_train.csv')

def build_model(x, y):
    return logistic_regression_522.get_model(x, y)

# def make_predictions():
#     pass 

def cross_validate():
    model = build_model(x, y)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    return scores 

def evaluate_predictions():
    def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
    def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
    def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
    def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
               'fp': make_scorer(fp), 'fn': make_scorer(fn)}

    print(scoring)
    
