import numpy as np

def myLDA(data, num_principal_components):

    X_train = data[:, :-1]
    y_train = data[:, -1]

    #class mean
    mean_vec = []
    for label in np.unique(y_train):
        mean_vec.append(np.mean(X_train[y_train == label]))
    mean_vec = np.array(mean_vec)

    #within-class scatter matrix
    WithinScat = np.zeros((X_train.shape[1], X_train.shape[1]))
    for label in np.unique(y_train):
        WithinScat += np.cov(X_train[y_train == label].T)

    #between-class scatter matrix
    BetweenScat = np.cov(mean_vec.T)

    #compute eigs
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(WithinScat).dot(BetweenScat))

    #sort eigs
    ind = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:,ind]
    eigvals = eigvals[ind]

    #projection matrix by selecting top pricomp
    proj_matrix = eigvecs[:, :num_principal_components]

    return proj_matrix, eigvals
