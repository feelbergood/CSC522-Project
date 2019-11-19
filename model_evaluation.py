import pandas as pd
import warnings
import sklearn
from classifiers.Baseline import BaselineModel
from classifiers.KNN import KNNModel
from classifiers.LogisticRegression import LRModel
from classifiers.svm_classifiers.SVM_C import SVMCModel
from classifiers.svm_classifiers.SVM_Nu import SVMNuModel
from classifiers.svm_classifiers.SVM_Linear import SVMLinearModel
from classifiers.decision_tree_classifiers.DecisionTree import DecisionTreeModel
from classifiers.emsemble_classifiers.AdaBoost import AdaBoostModel
from classifiers.emsemble_classifiers.Bagging import BaggingModel
from classifiers.emsemble_classifiers.RandomForest import RandomForestModel
from classifiers.bayes_classifiers.GaussianNB import GaussianNBModel
from classifiers.bayes_classifiers.BernoulliNB import BernoulliNBModel
from classifiers.bayes_classifiers.MultinomialNB import MultinomialNBModel
from classifiers.bayes_classifiers.ComplementNB import ComplementNBModel
from classifiers.neural_network_classifiers.MLPModel import MLPModel

warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

baseline = BaselineModel()
knn = KNNModel()
lr = LRModel()
svm_c = SVMCModel()
svm_nu = SVMNuModel()
svm_linear = SVMLinearModel()
dt = DecisionTreeModel()
adaboost = AdaBoostModel()
bagging = BaggingModel()
rf = RandomForestModel()
gaussian_nb = GaussianNBModel()
bernoulli_nb = BernoulliNBModel()
multi_nb = MultinomialNBModel()
complement_nb = ComplementNBModel()
mlp = MLPModel()

no_tuning_models = [baseline, knn, lr, svm_c, svm_nu, svm_linear, dt, adaboost, bagging, rf, gaussian_nb,
                    bernoulli_nb, multi_nb, complement_nb, mlp]


# Preprocessed Data 1: Dropping columns with missing data (1976-2004)
def get_xy():
    data = pd.read_csv('preprocessed_data/team_seasons_classified_1.csv')
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return x, y


# Preprocessed Data 2: Other ways of dealing with missing data
def get_xy_1():
    return 1, 1


# Experiment with data 1 and no hyper-parameter tuning
def no_tuning_experiment():
    x, y = get_xy()
    for model in no_tuning_models:
        model.get_and_save_performance(x, y, "results/train_evaluation_no_tuning.csv")
        print("--------------" + model.get_name() + "--------------")
        print(model.get_confusion_matrix())
        print(model.get_performance())


# Experiment with data 1 and hyper-parameter tuning
def tuning_experiment():
    print("Experiment with Hyper-parameter Tuning")


def main():
    print("1. Dropping columns with missing data (1976-2004)")
    print("1.1. Experiment without Hyper-parameter Tuning")
    no_tuning_experiment()
    print("\n1.2. Experiment with Hyper-parameter Tuning")
    tuning_experiment()


main()
