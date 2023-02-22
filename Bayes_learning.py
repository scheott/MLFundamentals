import numpy as np

def Bayes_Learning(training_data, validation_data):

    class1 = training_data[np.where(training_data[:, -1] == 1)][:, :-1]
    class2 = training_data[np.where(training_data[:, -1] == 2)][:, :-1]

    #Bernoulli parameters for each sum(x^t)/ N
    p1 = np.mean(class1, axis=0)
    p2 = np.mean(class2, axis=0)

    error_rates = []
    sigmas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6]
    for sigma in sigmas:
        pc1 = 1 - np.exp(-sigma) #prior function given
        pc2 = 1 - pc1

        # classify validation set using learned parameters and prior
        predicted_class = []
        for i in range(len(validation_data)):#Bayes thm
            posterior1 = np.prod(p1 ** validation_data[i, :-1] * (1 - p1) ** (1 - validation_data[i, :-1]))* pc1 / (np.prod(p1 ** validation_data[i, :-1] * (1 - p1) ** (1 - validation_data[i, :-1]))* pc1 + np.prod(p2 ** validation_data[i, :-1] * (1 - p2) ** (1 - validation_data[i, :-1]))* pc2)
            posterior2 = np.prod(p2 ** validation_data[i, :-1] * (1 - p2) ** (1 - validation_data[i, :-1]))* pc2 / (np.prod(p1 ** validation_data[i, :-1] * (1 - p1) ** (1 - validation_data[i, :-1]))* pc1 + np.prod(p2 ** validation_data[i, :-1] * (1 - p2) ** (1 - validation_data[i, :-1]))* pc2)
            #above is the likelyhood multiplied by prior
            if posterior1 > posterior2:
                predicted_class.append(1)
            else:
                predicted_class.append(2)
            #if p(c1) > p(c2) then x^t classifies as c1 and vice versa

        #below calcultes the error for each sigma
        error_rate = np.mean(predicted_class != validation_data[:, -1])
        error_rates.append(error_rate)
        print("Sigma: {}, Error Rate: {}".format(sigma, error_rate))

    # below chooses the best prior and portrays its error
    best_sigma = sigmas[np.argmin(error_rates)]
    pc1 = 1 - np.exp(-best_sigma)
    pc2 = 1 - pc1
    print("Best Sigma: {}".format(best_sigma))
    print(pc2)
    return p1, p2, pc1, pc2