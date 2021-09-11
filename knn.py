import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X = np.load('mnist_data.npy')
y = np.load('mnist_labels.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#model = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)

#print("Train Accuracy: ", metrics.accuracy_score(y_train, model.predict(X_train)))
#print("Test Accuracy: ", metrics.accuracy_score(y_test, model.predict(X_test)))


def k_plot(X_train, y_train, X_test, y_test):
    """ plots varying k values to find the best k"""
    plt.xlabel('Training Size') 
    plt.ylabel('Test Accuracy')
    plt.title('kNN Accuracy Analysis')
    test = []
    image_num = [3000, 6000, 9000, 12000, 15000, 18000, 21000]
    for num in image_num:
        model = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
        test.append(metrics.accuracy_score(y_test, model.predict(X_test)))
    plt.plot(image_num, test)
    plt.show()
      
#k_plot(X_train, y_train, X_test, y_test)

def trainingsize_plot(X_train, y_train, X_test, y_test):
    """ plots varying training sizes and compares accuracy """
    plt.xlabel('Training Size') 
    plt.ylabel('Test Accuracy')
    plt.title('kNN Accuracy Analysis')
    test = []
    image_num = [3000, 6000, 9000, 12000, 15000, 18000, 21000]
    for num in image_num:
        model = KNeighborsClassifier(n_neighbors=1).fit(X_train[0:num], y_train[0:num])
        test.append(metrics.accuracy_score(y_test, model.predict(X_test)))
    plt.plot(image_num, test)
    plt.show()
    
trainingsize_plot(X_train, y_train, X_test, y_test)