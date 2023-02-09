import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

def MyPerceptron(X,y, w0):

    w = w0
    a = np.linspace(min(X[:,0]), max(X[:,0]))
    # Compute the y-axis vector
    b = -w[0]/w[1] * a


    # Plot the data with different colors for different classes
    plt.scatter(X[y==1, 0], X[y==1, 1], c='r')
    plt.scatter(X[y==-1, 0], X[y==-1, 1], c='b')
    
    # Plot the line
    plt.plot(a, b)

    # Set the range of values for the x-axis and y-axis
    plt.xlim(min(X[:,0]) + .25, max(X[:,0]) + .25)
    plt.ylim(min(X[:,1]) + .25, max(X[:,1]) + .25)
    plt.title("Initialization")
    plt.show()

    steps = 0
    i = 1
    while i == 1:

        for i in range(40):

            dot_product = np.dot(w, X[i]) * y[i]
            # If the dot product is less than or equal to 0, update w

            if dot_product <= 0:
                steps += 1
                w = w + y[i] * X[i]

                a = np.linspace(min(X[:,0]), max(X[:,0]), 100)
                plt.scatter(X[y==1, 0], X[y==1, 1], c='r')
                plt.scatter(X[y==-1, 0], X[y==-1, 1], c='b') 
                b = -w[0]/w[1] * a
                plt.plot(a, b)
                plt.xlim(min(X[:,0]), max(X[:,0]))
                plt.ylim(min(X[:,1]), max(X[:,1]))
                plt.title("Converged")
                plt.show()
        i = i + 1


    return print(f"# of steps =  {steps} & w = [{w[0]}, {w[1]}]")


if __name__ == '__main__':
    w = [1, -1]
    w = np.array(w)
    mat_contents = sio.loadmat("data1.mat")
    X = mat_contents["X"]
    y = np.squeeze(mat_contents["y"])

    MyPerceptron(X,y, w)


