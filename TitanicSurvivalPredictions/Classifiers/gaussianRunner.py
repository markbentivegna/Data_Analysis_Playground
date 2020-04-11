import numpy as np
import pandas as pd
import itertools
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing import preprocessing_array, preprocessing_df
from selectkbest import select_k_best
from quartile_plot import generate_quartile_plot
from feature_importances import feature_importance_scores
from correlation_features import generate_correlation_matrix
from correlation_sort import sorted_correlation
from covariance_sort import sorted_covariance
import statistics

#FETCH AND PROCESS DATAFRAME
merge_filename = "data/merge.csv"
merge_df = pd.read_csv(merge_filename, encoding = "ISO-8859-1")
merge_df = preprocessing_df(merge_df)

#TEST SIZE
test_size = np.linspace(0.1,1,9,endpoint=False)

#GNB PARAMTERS 0.7646310432569975
var_smoothing = [1e-09, 1e-06, 1e-03, 1, 10]

#CLASSIFIER RESULTS CSV COLUMNS
test_size_list = []
smoothing_list = []
score_list = []

difference_list = []
std_list = []
mean_list = []

merge_df.dropna(subset=['Age'], inplace=True)

# transpose_x = np.transpose(merge_df.drop('Age', axis=1))
# sorted_covariance = sorted_covariance(transpose_x, merge_df.Age)
# sorted_correlation = sorted_correlation(transpose_x, merge_df.Age)
# quartile_data = [survived_df.Age, died_df.Age, survived_df.Fare, died_df.Fare, survived_df.Cabin, died_df.Cabin]
# generate_quartile_plot(quartile_data)


#selected_features = ["Pclass", "Sex", "Fare", "Embarked", "Age"]
selected_features = ["Pclass", "Parch", "SibSp", "Cabin"]
label="Age"
merge_df[label] = merge_df[label].astype('int')
for val, smoothing in itertools.product(test_size, var_smoothing):
    X_train, X_test, y_train, y_test = train_test_split(np.nan_to_num(merge_df[selected_features]), merge_df[label], test_size=val, random_state=1)
    
    clf = GaussianNB(var_smoothing=smoothing)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    test_size_list.append(val)
    smoothing_list.append(smoothing)
    score_list.append(accuracy_score(pred, y_test))

    i = 0
    for val in y_test:
        difference_list.append(val - pred[i])
        i+= 1
    mean_list.append(statistics.mean(difference_list))
    std_list.append(statistics.stdev(difference_list))
    

data={'Test Size': test_size_list, 'Smoothing Parameter': smoothing_list, 'Score': score_list, 'STDEV': std_list, 'MEAN': mean_list}
df = pd.DataFrame(data=data)
df.to_csv('results/gaussianScore.csv', index=False)
    