# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


def prepare_data_to_train(
    transactions_data_p: pd.DataFrame,
    user_events_data_p: pd.DataFrame,
    labels_set: pd.DataFrame,
) -> pd.DataFrame:
    """
    This function prepare data to train and test

    :param transactions_data_p: transactions data
    :param user_events_data_p: user events data
    :param labels_set: labels data
    :return: train and test data
    """
    transactions_data_p.drop(columns=["age", "target", "device"], inplace=True)
    transactions_data_p.sort_values(by="timestamp", inplace=True)
    user_events_data_p.sort_values(by="timestamp", inplace=True)

    merged_df = pd.merge_asof(
        transactions_data_p,
        user_events_data_p,
        on="timestamp",
        by="source",
    )

    data_for_train = (
        pd.merge_asof(merged_df, labels_set, on="timestamp", by="source")
        .drop(columns=["source", "timestamp"])
        .dropna()
    )

    lable = data_for_train["label"]
    data_for_train.drop(columns=["label"], inplace=True)

    return train_test_split(data_for_train, lable, test_size=0.2, random_state=42)


def train_and_val(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> RandomForestClassifier:
    """
    This function train and validate the model

    :param X_train: train data
    :param X_test: test data
    :param y_train: train labels
    :param y_test: test labels
    :return: model
    """
    grid_search = {
        "bootstrap": [True, False],
        "max_depth": [
            10,
            30,
            50,
            100,
        ],
        "max_features": ["log2", "sqrt"],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "n_estimators": [50, 100, 500],
    }

    rf = RandomForestClassifier()
    rfc = RandomizedSearchCV(
        estimator=rf,
        param_distributions=grid_search,
        n_iter=100,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    rfc.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rfc.best_estimator_.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print the evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    return rfc.best_estimator_
