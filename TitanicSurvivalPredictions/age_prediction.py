import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from selectkbest import select_k_best
from quartile_plot import generate_quartile_plot
from feature_importances import feature_importance_scores
from correlation_features import generate_correlation_matrix
from correlation_sort import sorted_correlation
from covariance_sort import sorted_covariance
from submission_csv import generate_csv_file
from sklearn import preprocessing

def age_classifier():
    merge_filename = "data/merge.csv"

    age_df = pd.read_csv(merge_filename, encoding = "ISO-8859-1")

    for col in age_df.select_dtypes(include=['object']).columns:
        age_df[col].fillna(value=0, inplace=True)
        age_df[col] = pd.Categorical(age_df[col], categories=age_df[col].unique()).codes
    y = age_df["Age"]
    age_df = age_df[age_df.Age.notna()]
    age_df["Age"] = age_df["Age"].astype('int')

    selected_features = ["Pclass", "Name", "SibSp", "Ticket", "Fare", "Cabin"]

    X_train, X_test, y_train, y_test = train_test_split(np.nan_to_num(age_df[selected_features]), age_df.Age, test_size=0.3, random_state=1)

    X_train_processed = np.nan_to_num(preprocessing.scale(X_train))
    X_test_processed = np.nan_to_num(preprocessing.scale(X_test))
    clf = MLPClassifier(solver='lbfgs', alpha=1e-10, hidden_layer_sizes=(10,2), random_state=1)
    clf.fit(X_train_processed, y_train)
    
    return clf
