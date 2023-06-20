import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

def prepare_data_to_train(transactions_data_p, user_events_data_p, labels_set):
    transactions_data_p.drop(columns=['age', 'target', 'device'], inplace=True)
    transactions_data_p.sort_values(by='timestamp', inplace=True)
    user_events_data_p.sort_values(by='timestamp', inplace=True)


    merged_df = pd.merge_asof(
        transactions_data_p,
        user_events_data_p,
        on='timestamp',
        by='source',
    )

    data_for_train = pd.merge_asof(
        merged_df,
        labels_set,
        on='timestamp',
        by='source'
    ).drop(columns=['source', 'timestamp']).dropna()
    
    lable = data_for_train['label']
    data_for_train.drop(columns=['label'], inplace=True)

    return train_test_split(data_for_train, lable, test_size=0.2, random_state=42)

def train_and_val(X_train, X_test, y_train, y_test):
    grid_search = {'bootstrap': [True, False],
                   'max_depth': [10, 30, 50, 100,],
                   'max_features': ['auto', 'sqrt'],
                   'min_samples_leaf': [1, 2, 4],
                   'min_samples_split': [2, 5, 10],
                   'n_estimators': [50, 100, 500]}

    rf = RandomForestClassifier()
    rfc = RandomizedSearchCV(estimator = rf, param_distributions = grid_search, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
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
