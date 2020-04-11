import numpy as np
import pandas as pd
import itertools
from sklearn import svm
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

#SVM PARAMTERS - 0.7820186598812553 (0)
C = [1e-09, 1e-06, 1e-03, 1, 10, 100]
kernel_string = ['rbf', 'linear', 'sigmoid']
#kernel_string = ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']
degreeint = np.linspace(1, 3, 3)
gamma = ['scale', 'auto']
decision_function_shape = ["ovo", "ovr"]

#CLASSIFIER RESULTS CSV COLUMNS
test_size_list = []
C_list = []
kernel_list = []
degree_list = []
gamma_list = []
decision_shape_list = []
score_list = []

difference_list = []
std_list = []
mean_list = []

merge_df.dropna(subset=['Age'], inplace=True)

#selected_features = ["Pclass", "Sex", "Fare", "Embarked", "Age"]
selected_features = ["Pclass", "Parch", "SibSp", "Cabin"]
label="Age"
merge_df[label] = merge_df[label].astype('int')

for val, reg_param, kernel, degree, reg_gamma, shape in itertools.product(test_size, C, kernel_string, degreeint, gamma, decision_function_shape):
    X_train, X_test, y_train, y_test = train_test_split(np.nan_to_num(merge_df[selected_features]), merge_df.Survived, test_size=val, random_state=1)
    
    clf = svm.SVC(C=reg_param, kernel=kernel, degree=degree, gamma=reg_gamma, decision_function_shape=shape, random_state=1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    test_size_list.append(val)
    C_list.append(reg_param)
    kernel_list.append(kernel)
    degree_list.append(degree)
    gamma_list.append(reg_gamma)
    decision_shape_list.append(shape)
    score_list.append(accuracy_score(pred, y_test))

    i = 0
    for val in y_test:
        difference_list.append(val - pred[i])
        i+= 1
    mean_list.append(statistics.mean(difference_list))
    std_list.append(statistics.stdev(difference_list))

data={'Test Size': test_size_list, 'C parameter': C_list, 'Kernel String': kernel_list, 'Degree Int': degree_list, 'Gamma': gamma_list, 'Decision Shape': decision_shape_list, 'Score': score_list, 'STDEV': std_list, 'MEAN': mean_list}
df = pd.DataFrame(data=data)
df.to_csv('results/svmScore.csv', index=False)

    