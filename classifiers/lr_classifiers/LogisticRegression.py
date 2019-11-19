import pandas as pd
import os
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from classifiers.Model import Model

dir_name = os.path.dirname(__file__)
data_path = os.path.join(dir_name, '../../preprocessed_data/team_seasons_classified_1.csv')


def get_tuned_parameters():
    param_grid = {"n_neighbors": range(1, 100),
                  "weights": ["uniform", "distance"],
                  "metric": ["euclidean", "manhattan", "minkowski"]}
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=10)
    data = pd.read_csv(data_path)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    grid_search.fit(x, y)
    print("Best parameters:{}".format(grid_search.best_params_))
    return grid_search.best_params_


class LRModel(Model):
    def __init__(self):
        self.name = "LR"
        self.model = LogisticRegression(solver='lbfgs')


# get_tuned_parameters()
