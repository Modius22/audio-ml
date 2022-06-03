from typing import bool

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def get_dataset(
    notebook: bool = False, smote: bool = False
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    if not notebook:
        train_dataset = pd.read_csv("data/data.csv")
    else:
        train_dataset = pd.read_csv("../data/data.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        train_dataset.drop(columns=["type"]), train_dataset["type"], test_size=0.33, random_state=42
    )

    if smote:
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    return X_train, y_train, X_test, y_test
