import csv 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

train_filename = "data/train.csv"
test_filename = "data/test.csv"

features_fields = ["Pclass", "Sex", "Parch", "Fare", "Cabin", "Embarked"]

training_fields = [] 
training_rows = []
label_field = "Survived"
label_index = 0
training_features_index = []
training_features = []
labels = []

test_fields = []
test_rows = []
test_features_index = []
test_features = []
passenger_id_list = []

string_to_int_mapping = {}

with open(train_filename, 'r') as csvfile: 
	csvreader = csv.reader(csvfile) 
	training_fields = csvreader.__next__()

	for row in csvreader: 
		training_rows.append(row) 
	print("Total no. of rows: %d"%(csvreader.line_num)) 
	print('Field names are:' + ', '.join(field for field in training_fields)) 

with open(test_filename, 'r') as csvfile: 
	csvreader = csv.reader(csvfile) 
	test_fields = csvreader.__next__()

	for row in csvreader: 
		test_rows.append(row) 
	print("Total no. of rows: %d"%(csvreader.line_num)) 
	print('Field names are:' + ', '.join(field for field in test_fields))

for i in range(len(training_fields)):
	if training_fields[i] == label_field:
		label_index = i
	if training_fields[i] in features_fields:
		training_features_index.append(i)

for i in range(len(test_fields)):
	if test_fields[i] in features_fields:
		test_features_index.append(i)

print('survived index:', label_index)

for row in training_rows:
	col_count = 0
	new_features = []
	for field in row:
		if col_count == label_index:
			labels.append(int(field))
		if col_count in training_features_index:
			try:
				new_features.append(float(field))
			except ValueError:
				try:
					if field[0] not in string_to_int_mapping:
						string_to_int_mapping[field[0]] = len(string_to_int_mapping) + 1
					new_features.append(string_to_int_mapping[field[0]])
				except IndexError:
					new_features.append(-1)
		col_count += 1
	training_features.append(new_features)

for row in test_rows:
	col_count = 0
	new_features = []
	for field in row:
		if col_count == 0:
			passenger_id_list.append(field)
		if col_count in test_features_index:
			try:
				new_features.append(float(field))
			except ValueError:
				try:
					if field[0] not in string_to_int_mapping:
						string_to_int_mapping[field[0]] = len(string_to_int_mapping) + 1
					new_features.append(string_to_int_mapping[field[0]])
				except IndexError:
					new_features.append(-1)
		col_count += 1
	test_features.append(new_features)


x = np.array(training_features)
x = x.astype(np.float64)
clf = GaussianNB()
clf.fit(x, labels)
pred = clf.predict(test_features)

pclass = []
sexes = []

for feature in training_features:
	pclass.append(feature[0])
	sexes.append(feature[1])
	
#The two features with the strongest correlation coefficients for survival are class and sex
class_coef = np.corrcoef(pclass, labels)
print("correlation coefficient by class:", class_coef)

sex_coef = np.corrcoef(sexes, labels)
print("correlation coefficient by sex:", sex_coef)

colors = []
for label in labels:
	if label == 1:
		colors.append('red')
	else:
		colors.append('blue')

with open ('submission.csv', 'w', newline='') as csvfile:
	fieldnames = ['PassengerId', 'Survived']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

	writer.writeheader()
	for i in range(len(pred)):
		writer.writerow({'PassengerId': passenger_id_list[i], 'Survived': pred[i]})
