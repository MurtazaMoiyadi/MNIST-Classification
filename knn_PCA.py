import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time


X = np.load('mnist_data.npy')
y = np.load('mnist_labels.npy')



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components = 200)
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

model = KNeighborsClassifier(n_neighbors=1).fit(X_train[:17500], y_train[:17500])

print("Train Accuracy: ", metrics.accuracy_score(y_train, model.predict(X_train)))
print("Test Accuracy: ", metrics.accuracy_score(y_test, model.predict(X_test)))

def plot(X_train, y_train, X_test, y_test):
    """ plots varying training sizes and compares accuracy """
    plt.xlabel('Components') 
    plt.ylabel('Time')
    plt.title('kNN Accuracy Analysis')
    times = []
    times2 = []
    image_num = [3000, 6000, 9000, 12000, 15000, 18000, 21000]
    pca_nums = [50, 150, 250, 350, 450, 550, 650, 750]
    for num in image_num:
        start_time = time.time()
        model = KNeighborsClassifier(n_neighbors=1).fit(X_train[0:num], y_train[0:num])
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
  
    for num in pca_nums:
        newX_train = X_train
        pca = PCA(n_components=num)
        pca.fit(newX_train)
        newX_train = pca.transform(X_train)
        start_time = time.time()
        model2 = KNeighborsClassifier(n_neighbors=1).fit(newX_train, y_train)
        elapsed_time = time.time() - start_time
        times2.append(elapsed_time)
    #plt.plot(image_num, times)
    plt.plot(pca_nums, times2).color("orange")
    plt.show()
    
