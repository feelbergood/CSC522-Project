import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Linear-Support Vector Classification
from classifiers.Model import Model


def get_model():
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(LinearSVC(), param_grid, cv=5)

    data = pd.read_csv('preprocessed_data/team_seasons_classified_1.csv')

    x = data[['o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_oreb',
              'o_dreb', 'o_reb', 'o_asts', 'o_pf', 'o_stl', 'o_to', 'o_blk', 'o_pts', 'd_fgm', 'd_fga', 'd_ftm',
              'd_fta', 'd_oreb',
              'd_dreb', 'd_reb', 'd_asts', 'd_pf', 'd_stl', 'd_to', 'd_blk', 'd_pts', 'pace']]
    y = data['class']
    mapper = DataFrameMapper([(x.columns, StandardScaler())])
    x = mapper.fit_transform(x, 4)
    mapper = DataFrameMapper([(y, LabelEncoder())])
    y = mapper.fit_transform(y, 4).ravel()

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=10)
    grid_search.fit(X_train, y_train)
    # print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
    # print("Best parameters:{}".format(grid_search.best_params_))
    # print("Best score on train set:{:.2f}".format(grid_search.best_score_))

    svm = LinearSVC(C=grid_search.best_params_.get("C"))
    return svm


class SVMLinearModel(Model):
    def __init__(self):
        self.name = "Linear-Support SVC"
        self.model = LinearSVC()
