import numpy as np

def myPCA(data, num_principal_components):

    #covariance matrix
    covar = np.cov(data.T)

    #calc eigs
    eigvals, eigvecs = np.linalg.eigh(covar)

    #sort eigs
    ind = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:,ind]
    eigvals = eigvals[ind]

    #form princicap components
    principal_components = eigvecs[:,:num_principal_components]

    return principal_components, eigvals
