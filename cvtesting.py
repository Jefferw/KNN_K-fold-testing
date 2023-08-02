from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def hyper_param_tuning_knn(dataset, knn_min_k, knn_max_k):
    """This function takes in a data set and trains a knn classifier model, testing k-values passed in at function call"""
    """Returns opimal value of k based on 10-fold cross validation, and the 4 split datasets"""
    X = pd.DataFrame(dataset.get('data'))
    Y = pd.DataFrame(dataset.get('target'))
    training_x, testing_x, training_y, testing_y = train_test_split(X, Y, test_size = 0.33, random_state=42)
    cv_k_values = list(range(knn_min_k, knn_max_k+1))
    cv_k_scores = []
    high_score = 0
    for k in cv_k_values:
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn_classifier, training_x, training_y.values.ravel(), cv=10, scoring='accuracy')
        #print("average accuracy on 10-fold cross validation for k="+str(k)+": " + str(np.mean(scores)))
        cv_k_scores.append(np.mean(scores))
        if np.mean(scores) > high_score:
            high_score = np.mean(scores)
            return_k = k
    plt.plot(cv_k_scores)
    plt.xticks(cv_k_values)
    plt.show()
    return return_k, training_x, testing_x, training_y, testing_y

dataset_iris = datasets.load_iris()

k, training_x, testing_x, training_y, testing_y = hyper_param_tuning_knn(dataset_iris, 1, 20)
trained_knn_classifier = KNeighborsClassifier(n_neighbors=k)
trained_knn_classifier.fit(training_x, training_y.values.ravel())
predictions = trained_knn_classifier.predict(testing_x)
print('Highest k-fold cross-validation at k =', k)
print('KNN accuracy score on test data with k =', k, ':', accuracy_score(testing_y, predictions))

