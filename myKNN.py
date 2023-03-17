import numpy as np

def myKNN(training_data, test_data, k):

    ##load data
    X_train = training_data[:, :-1]
    y_train = training_data[:, -1]
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    ##create array of predictions
    preds = np.zeros(len(X_test))

    for i in range(len(X_test)):
        distances = np.zeros(len(X_train))

        for j in range(len(X_train)):
            distances[j] = np.sqrt(np.sum((X_test[i] - X_train[j])**2)) ##euclidean distance

        ksort = np.argsort(distances) #sorting distances on indices
        k_ind = ksort[:k] ##takes first k elements
        labels = y_train[k_ind]

        preds[i] = np.argmax(np.bincount(labels.astype(int))) ##predicting label of point, taking most common label of knn

    acc = np.mean(preds == y_test) 

    print(f"Accuracy: {acc:.4f}")
    return preds