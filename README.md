# MNIST-Classification
## Implements K-Nearest-Neighbors and logistic regression to perform classification on the MNIST hand-drawn numbers dataset. Then uses PCA decomposition to reduce the dimensionality of the dataset and compares the performance of KNN with varying component and training sizes.

*knn.py* - implements the KNeighborsClassifer algorithm via Sklearn and plots the accuracy score of the algorithm based on varying K inputs and varying training set sizes.
  - While obvious, the larger the training size the higher the test accuracy.
  - Curiously, the larger K is, the lower the test accuracy.

*logisticregression.py* - implements multinomial logistic regression on the dataset via Sklearn.
  - Achieved a test accuracy of 0.8893.

*PCA_decomp.py* - utilizes StandardScaler and PCA via and MatPlotLib to plot and determine the optimal number of principal components while maintaining a reasonable percentage of the explained variance. 
  - Based on the resulting Cumulative Explained Variance plot, I chose 200 as the number of principal components in my decomposition algorithm.

*knn_PCA.py* - tests how varying the dimensions of the dataset and the training size affects training accuracy and performance of the KNN algorithm.
    - Having over 700 components takes almost 8x longer than the optimal number of 200.
    - As expected, training size directlty impacts performance speed


[MNIST dataset can be downloaded here.](http://yann.lecun.com/exdb/mnist/)
