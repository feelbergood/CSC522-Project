import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import pandas as pd

class Model:
    def __init__(self):
        self.name = "Model"
        self.model = None
        self.confusion_matrix = None
        self.performance = None

    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.confusion_matrix = None
        self.performance = None

    def get_name(self):
        return self.name

    def get_model(self):
        return self.model

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def get_performance(self):
        return self.performance

    def save_performance(self, filename):
        if os.path.isfile(filename):
            evaluation_metrics = pd.read_csv(filename, index_col=0)
            new_row = self.performance
            self.performance = new_row
            evaluation_metrics.loc[self.name] = new_row
            evaluation_metrics.to_csv(filename)
        else:
            new_row = self.performance
            self.performance = new_row
            df = pd.DataFrame.from_dict({self.name:self.performance}, orient="index")
            df.to_csv(filename)

    def get_and_save_performance(self, x, y, filename):
        y_pred = cross_val_predict(self.model, x, y, cv=10)
        self.confusion_matrix = confusion_matrix(y, y_pred)

        def tn(y, y_pred): return confusion_matrix(y, y_pred)[0, 0]

        def fp(y, y_pred): return confusion_matrix(y, y_pred)[0, 1]

        def fn(y, y_pred): return confusion_matrix(y, y_pred)[1, 0]

        def tp(y, y_pred): return confusion_matrix(y, y_pred)[1, 1]

        tp, tn, fp, fn = int(tn(y, y_pred)), int(tp(y, y_pred)), int(fn(y, y_pred)), int(fp(y, y_pred))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision) * (recall) / (precision + recall)
        perf = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": accuracy, "precision": precision, "recall": recall,
                "f1": f1}
        self.performance = perf
        self.save_performance(filename)
        return perf
