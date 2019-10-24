# CSC522 Project
NBA teams making playoffs prediction

## Work flow for adding new models
- Libraries: pandas, sklearn, sklearn-pandas

- Create your own model by creating a new python file with two methods get_model() and get_name(), for example:
```{python}
def get_model():
    logisticRegr = LogisticRegression()
    # model = logisticRegr.fit()
    return logisticRegr

def get_name():
    return "LR"
```

- import your model by importing ```<modelname>.py``` you created in the file ```model_evaluation.py```
- add your model to ```models``` list in ```model_evaluation.py```
- run ```model_evaluation.py``` and see the results in ```train_evaluations.csv```