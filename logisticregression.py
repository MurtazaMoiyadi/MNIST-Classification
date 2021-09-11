import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

X = np.load('mnist_data.npy')
y = np.load('mnist_labels.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X_train, y_train)
print("Train Accuracy: ", metrics.accuracy_score(y_train, mul_lr.predict(X_train)))
print("Test Accuracy: ", metrics.accuracy_score(y_test, mul_lr.predict(X_test)))