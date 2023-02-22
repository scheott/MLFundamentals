import numpy as np

def Bayes_Testing(test_data, p1, p2, pc1, pc2):
# classify test set using learned parameters and best prior
    predicted_class = []
    for i in range(len(test_data)):#Bayes thm
        posterior1 = np.prod(p1 ** test_data[i, :-1] * (1 - p1) ** (1 - test_data[i, :-1]))* pc1 / (np.prod(p1 ** test_data[i, :-1] * (1 - p1) ** (1 - test_data[i, :-1]))* pc1 + np.prod(p2 ** test_data[i, :-1] * (1 - p2) ** (1 - test_data[i, :-1]))* pc2)
        posterior2 = np.prod(p2 ** test_data[i, :-1] * (1 - p2) ** (1 - test_data[i, :-1]))* pc2 / (np.prod(p1 ** test_data[i, :-1] * (1 - p1) ** (1 - test_data[i, :-1]))* pc1 + np.prod(p2 ** test_data[i, :-1] * (1 - p2) ** (1 - test_data[i, :-1]))* pc2)
        #above is the likelyhood multiplied by the bern param
        if posterior1 > posterior2:
            predicted_class.append(1)
        else:
            predicted_class.append(2)
            #if p(c1) > p(c2) then x^t classifies as c1 and vice versa

    # calculate error rate on test set using best prior
    error_rate = np.mean(predicted_class != test_data[:, -1])
    print("Error on the test set: {}".format(error_rate))