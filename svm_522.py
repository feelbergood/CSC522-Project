import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def get_model():
    # C：惩罚项，float类型，可选参数，默认为1.0，C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低，也就是对测试数据的分类准确率降低。相反，减小C的话，容许训练样本中有一些误分类错误样本，泛化能力强。对于训练样本带有噪声的情况，一般采用后者，把训练样本集中错误分类的样本作为噪声。
    # 核函数系数，float类型，可选参数，默认为auto。只对’rbf’ ,’poly’ ,’sigmod’有效。如果gamma为auto，代表其值为样本特征数的倒数，即1/n_features
    param_grid = {"gamma": [0.001, 0.01, 0.1, 1, 10, 100],
                  "C": [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)  # 实例化一个GridSearchCV类

    data = pd.read_csv('output/team_seasons_classified_1_train.csv')

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
    grid_search.fit(X_train, y_train)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
    # print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
    # print("Best parameters:{}".format(grid_search.best_params_))
    # print("Best score on train set:{:.2f}".format(grid_search.best_score_))

    svm = SVC(gamma=grid_search.best_params_.get("gamma"), C=grid_search.best_params_.get("C"))
    return svm


def get_name():
    return "SVM"
