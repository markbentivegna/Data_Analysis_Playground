import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing import preprocessing_array, preprocessing_df
from selectkbest import select_k_best
from quartile_plot import generate_quartile_plot
from feature_importances import feature_importance_scores
from correlation_features import generate_correlation_matrix
from correlation_sort import sorted_correlation
from covariance_sort import sorted_covariance
from submission_csv import generate_csv_file

train_filename = "data/train.csv"
test_filename = "data/test.csv"
train_df = pd.read_csv(train_filename, encoding = "ISO-8859-1")
test_df = pd.read_csv(test_filename, encoding = "ISO-8859-1")
train_df = preprocessing_df(train_df)
test_df = preprocessing_df(test_df)

transpose_x = np.transpose(train_df.drop('Survived', axis=1))
sorted_covariance = sorted_covariance(transpose_x, train_df.Survived)
sorted_correlation = sorted_correlation(transpose_x, train_df.Survived)

k_best_features = select_k_best(train_df, "Survived")
# feature_importance_scores = feature_importance_scores(train_df)
# generate_correlation_matrix(train_df)
# quartile_data = [survived_df.Age, died_df.Age, survived_df.Fare, died_df.Fare, survived_df.Cabin, died_df.Cabin]
# generate_quartile_plot(quartile_data)

selected_features = ["Pclass", "Sex", "Fare", "Cabin", "Embarked", "Age"]
y_train = train_df.Survived
X_train, X_test = preprocessing_array(train_df, test_df, selected_features)
# X_train, X_test, y_train, y_test = train_test_split(np.nan_to_num(train_df[selected_features]), train_df.Survived, test_size=0.3)

# clf = DecisionTreeClassifier()
# clf = GaussianNB()
# clf = svm.SVC()
#clf = AdaBoostClassifier(n_estimators=1000)
clf = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(5,2), random_state=1)

clf.fit(X_train, y_train)
print("TRAINING SCORE:", clf.score(X_train, y_train))
pred = clf.predict(X_test)

generate_csv_file(test_df, pred, 'submission.csv')
