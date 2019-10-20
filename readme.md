# CSC522_Project
Project for CSC522: Automated Learning and Data Analysis




# work flow:
1. pip install sklearn-pandas

2. add your own model into the root folder, refer to the file logistic_regression_522.py, create a function that outputs the model:
```{python}
def get_model():
    logisticRegr = LogisticRegression()
    # model = logisticRegr.fit()
    return logisticRegr
```
3. import your model by writing import <modelname>.py at the top of the file model_evaluation.py
4. change the function in build_model()
5. in evaluate_prediction, change the model name to your own model name at around line 40
5. run runme()