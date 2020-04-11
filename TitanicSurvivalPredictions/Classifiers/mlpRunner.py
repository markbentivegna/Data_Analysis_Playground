import numpy as np
import pandas as pd
import itertools
from sklearn.neural_network import MLPClassifier
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

max_score = 0
#TEST SIZE
test_size = np.linspace(0.1,1,9,endpoint=False)

#MLP Params - 0.7938931297709924 (predicted)
solver_list = ['lbfgs', 'sgd', 'adam']
alpha_list = [1e-09, 1e-06, 1e-03]
x_layers = np.linspace(6,10, 5)
y_layers = np.linspace(1,5, 5)

test_size_list = []
solvers_list = []
alphas_list = []
hidden_layer_size = []
score_list = []

difference_list = []
std_list = []
mean_list = []

merge_df.dropna(subset=['Age'], inplace=True)

#selected_features = ["Pclass", "Sex", "Fare", "Embarked", "Age"]
selected_features = ["Pclass", "Parch", "SibSp", "Cabin"]
label="Age"
merge_df[label] = merge_df[label].astype('int')

for val, solver, alpha, x, y in itertools.product(test_size, solver_list, alpha_list, x_layers, y_layers):
    X_train, X_test, y_train, y_test = train_test_split(np.nan_to_num(merge_df[selected_features]), merge_df[label], test_size=val, random_state=1)

    clf = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=(int(x),int(y)), random_state=1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    test_size_list.append(val)
    solvers_list.append(solver)
    alphas_list.append(alpha)
    hidden_layer_size.append('( ' + str(x) + ',' + str(y) + ')')
    score_list.append(accuracy_score(pred, y_test))

    i = 0
    for val in y_test:
        difference_list.append(val - pred[i])
        i+= 1
    mean_list.append(statistics.mean(difference_list))
    std_list.append(statistics.stdev(difference_list))
    
data={'Test Size': test_size_list, 'Solver': solvers_list, 'Alpha':alphas_list, 'Hidden Layer Size': hidden_layer_size, 'Score': score_list, 'STDEV': std_list, 'MEAN': mean_list}
df = pd.DataFrame(data=data)
df.to_csv('results/mlpScore.csv', index=False)
    