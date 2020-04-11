import numpy as np
import pandas as pd
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing import preprocessing_array, preprocessing_df
import statistics

#FETCH AND PROCESS DATAFRAME
merge_filename = "data/merge.csv"
merge_df = pd.read_csv(merge_filename, encoding = "ISO-8859-1")
merge_df = preprocessing_df(merge_df)

#TEST SIZE
test_size = np.linspace(0.1,1,9,endpoint=False)

#DT PARAMTERS - 0.7984732824427481 (0)
criterion = ["gini", "entropy"]
splitter = ["best", "random"]
max_depth = np.linspace(5, 35, 31)

test_size_list = []
criterion_list = []
splitter_list = []
max_depth_list = []
score_list = []

difference_list = []
std_list = []
mean_list = []

merge_df.dropna(subset=['Age'], inplace=True)

#selected_features = ["Pclass", "Sex", "Fare", "Embarked", "Age"]
selected_features = ["Pclass", "Parch", "SibSp", "Cabin"]
label="Age"
merge_df[label] = merge_df[label].astype('int')

for val, criteria, split, depth in itertools.product(test_size, criterion, splitter, max_depth):
    X_train, X_test, y_train, y_test = train_test_split(np.nan_to_num(merge_df[selected_features]), merge_df[label], test_size=val, random_state=1)
    
    clf = DecisionTreeClassifier(criterion = criteria, splitter = split, max_depth=int(depth), random_state = 1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    test_size_list.append(val)
    criterion_list.append(criteria)
    splitter_list.append(split)
    max_depth_list.append(depth)
    score_list.append(accuracy_score(pred, y_test))

    i = 0
    for val in y_test:
        difference_list.append(val - pred[i])
        i+= 1
    mean_list.append(statistics.mean(difference_list))
    std_list.append(statistics.stdev(difference_list))

data={'Test Size': test_size_list, 'Criterion': criterion_list, 'Splitter': splitter_list, 'Max Depth': max_depth_list, 'Score': score_list, 'STDEV': std_list, 'MEAN': mean_list}
df = pd.DataFrame(data=data)
df.to_csv('results/decisionTreeScore.csv', index=False)
    