import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from skimage import io
import matplotlib.pyplot as plt

def EMG(image, k, flag):
    #normalize pixel values to be between 0 and 1
    img = io.imread(image)
    img = img/255
    
    r, c, d = img.shape
    X = img.reshape((r*c, d))

    #initialize means using k-means
    kmeans = KMeans(n_clusters=k, n_init=3, max_iter=3).fit(X)
    labs = kmeans.predict(X) #get labels for each data point
    mu = kmeans.cluster_centers_[labs] #initialize means

    #covariance matrix for each cluster
    sigma = []
    for i in range(k):
        X_cluster = X[labs == i]
        sigma_i = np.cov(X_cluster.T)
        sigma.append(sigma_i)
    sigma = np.array(sigma)

    pi = np.ones(k) / k #weights
    h = np.zeros((r*c, k)) #responsibilities matrix
    
    if flag == 0:
        Qm = np.zeros(100)
    Qe = np.zeros(100)
    for t in range(100):
        #e-step
        for i in range(k):
            #calc the responsibility of data points
            h[:,i] = pi[i] * multivariate_normal.pdf(X, mean=mu[i], cov=sigma[i]) 

        h /= h.sum(axis=1, keepdims=True) #normalize responsibilities
        
        Qe[t] = 0
        for i in range(k):
            Qe[t] += (h[:,i] * np.log(pi[i]) + h[:,i] * multivariate_normal(mean=mu[i], cov=sigma[i]).logpdf(X)).sum()

        #m-step 
        s = h.sum(axis=0)
        pi = s / (r*c) #update mixture weights
        mu = np.array([np.sum(h[:, i][:, np.newaxis] * X, axis=0) / s[i] for i in range(k)]) #update means
        sigma = []
        for i in range(k):
            X_cluster = X[kmeans.labels_ == i]
            sigma_i = np.cov(X_cluster.T)
            sigma.append(sigma_i)
        sigma = np.array(sigma) #update covariances

        Qm_prev = 0
        
        if flag == 0:
            #compute log-likelihood after M step
            Qm[t] = 0
            for i in range(k):
                Qm[t] += (h[:,i] * np.log(pi[i]) + h[:,i] * multivariate_normal(mean=mu[i], cov=sigma[i]).logpdf(X)).sum()

            #converges if difference between Qm in the current and previous iteration is small enough
            if t > 0 and abs(Qm[t]-Qm[t-1]) < 1e-6 * abs(Qm[t]):
                break

        else:
            lamda = 5
            #same as the orginial function with the regularization term
            Qm = 0
            for i in range(k):
                Qm += np.sum(h[:,i] * np.log(pi[i]) + h[:,i] + multivariate_normal.logpdf(X, mean=mu[i], cov=sigma[i])) - np.eye(d) * lamda/2
                
            if t > 0 and abs(Qm-Qm_prev).all() < .000001 * abs(Qm).all(): 
                break

            Qm_prev = Qm

    compressed_img = np.zeros_like(X) 
    for i in range(k):
        #pixels that belong to the i-th cluster and set their values to the mean of the cluster
        compressed_img[np.argmax(h, axis=1) == i] = mu[i]

    #reshape the compressed image
    compressed_img = compressed_img.reshape(r, c, d)

    #Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(compressed_img)
    ax1.set_title(f'k = {k} Compressed Image')

    ax2.plot(Qe, label="after e step")
    ax2.plot(Qm, label="after m step") 
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Expected Complete Log-Likelihood')
    ax2.set_title(f'k = {k}')
    plt.ylim(0)
    plt.legend()
    plt.show()

    return h, mu, Qm[:t]