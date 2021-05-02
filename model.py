import os

import numpy as np
import pandas as pd

from preprocessing import DataProcessor, get_main_variables

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def build_pipe():

    ct_num = ColumnTransformer(transformers = [
        ("impute", SimpleImputer(), ["Age", "Fare"]),
        # ("impute_cat", SimpleImputer(strategy = "most_frequent"), ["Embarked"]),
        ("ohe", OneHotEncoder(), ["Sex"])
        ], remainder = "passthrough"
    )

    pipe = Pipeline(steps = [
        ("ct_pipe", ct_num),
        ("model", LogisticRegression())
    ])

    return pipe

def create_submission_file(pipe, X_test):
    y_pred = pipe.predict(X_test)
    X_test["Survived"] = y_pred
    X_test = X_test[["Survived"]]
    X_test.to_csv("submission.csv")

def main():
    PATH_TRAIN, PATH_TEST, ID_COLUMN, TRAIN_COLUMNS, TARGET_COLUMN = get_main_variables()

    dp = DataProcessor(PATH_TRAIN, PATH_TEST, ID_COLUMN, TRAIN_COLUMNS, TARGET_COLUMN)

    X_train, y, X_test = dp.process_data()

    pipe = build_pipe()

    cross_val_score = round(np.mean(cross_validate(pipe, X_train, y, scoring = ["accuracy"], cv = 5)["test_accuracy"]), 3)
    print(f"Cross Validation Score is {cross_val_score}")
    
    pipe.fit(X_train, y)
    create_submission_file(pipe, X_test)

if __name__ == "__main__":
    main()