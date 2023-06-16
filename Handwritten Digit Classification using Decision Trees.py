import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn import preprocessing

train_data = pd.read_csv("/kaggle/input/handwritten-digit-identification-with-dt-b2/train.csv", header=None)
train_data.head()

X = train_data[train_data.columns[0:65]]
X

y = train_data[train_data.columns[65]]
y

X_train, X_test = X[:1143], X[1143:]
y_train, y_test = y[:1143], y[1143:]

mm_scaler = preprocessing.MinMaxScaler()

X_train_minmax = mm_scaler.fit_transform(X_train)

X_test_minmax = mm_scaler.fit_transform(X_test)

clf = DecisionTreeClassifier()

clf = clf.fit(X_train_minmax, y_train)

y_pred = clf.predict(X_test_minmax)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

train_data = pd.read_csv("/kaggle/input/handwritten-digit-identification-with-dt-b2/train.csv", header=None)
test_data = pd.read_csv("/kaggle/input/handwritten-digit-identification-with-dt-b2/test.csv", header=None)

X_train = train_data[train_data.columns[0:65]]

y_train = train_data[train_data.columns[65]]

X_test = test_data[test_data.columns[65]]

X_train_minmax = mm_scaler.fit_transform(X_train)

X_test_minmax = mm_scaler.fit_transform(X_test)

clf = clf.fit(X_train_minmax, y_train)

y_test = clf.predict(X_test_minmax)

output = pd.DataFrame({'ID':test_data[0], 'Category':y_test})
output.to_csv('submission.csv', index=False)
print("Submitted!")
