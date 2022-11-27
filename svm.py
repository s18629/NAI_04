import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import svm, metrics

#Load data without headers.
sonar_data = pd.read_csv('sonar.csv', header=None)

print(sonar_data)

D = sonar_data.values


"""
    Split the data in training and testing subsets.
"""
x = D[:,0:59]
y = D[:,60]

x_training, x_testing, y_training, y_testing = train_test_split(x, y, test_size=0.20)

model = SVC()
model.fit(x_training, y_training)

predict_sonar = model.predict(x_testing)

print("Accuracy: ", accuracy_score(y_testing, predict_sonar))

svc = svm.SVC(kernel='poly')
svc.fit(x_training, y_training)

y_model_outcome = svc.predict(x_testing)
print(f"model:  {y_model_outcome[0:10]}")
print(f"goal: {y_testing[0:10]}")

score = metrics.f1_score(y_testing, y_model_outcome, average=None)
print(score)

