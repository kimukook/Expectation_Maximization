'''
This is a simple script that fits a Multivariate Gaussian Model to the observed data.
=========================
Author  :  Muhan Zhao

Date    :  Aug. 17, 2019

Location:  West Hill, LA, CA
=========================
'''
import numpy as np
import matplotlib.pyplot as plt


class MultiGaussian:
    def __init__(self, data):
        # load the data
        self.data = data
        # initialize the size
        self.n = data.shape[0]
        self.m = data.shape[1]
        # initialize the parameters
        self.mu = np.zeros((self.n, 1))
        self.sigma = np.zeros((self.n, self.n))
        # determine the minimum and maximum range of each coordinates
        self.range = np.zeros((self.n, 2))
        # the first column is the minimum - 0.1 * range
        # the second column is the maximum + 0.1 * range
        for i in range(self.n):
            coordinate_range = np.max(self.data[i, :]) - np.min(self.data[i, :])
            self.range[i, 0] = np.min(self.data[i, :]) - .1 * coordinate_range
            self.range[i, 1] = np.max(self.data[i, :]) + .1 * coordinate_range

    def fit(self):
        '''
        MLE method to update mean and covariance of Gaussian model
        :return:
        '''
        self.mu = np.mean(self.data, axis=1).reshape(-1, 1)
        for i in range(self.m):
            current_data = self.data[:, i].reshape(-1, 1)
            self.sigma += np.dot(current_data - self.mu, (current_data - self.mu).T)
        self.sigma /= self.m

    def plot2D(self):
        '''
        2D plot function
        :return:
        '''
        assert self.data.shape[0] == 2, 'This is a 2D data visualizer'
        plt.figure()
        # scatter plot the data
        plt.scatter(self.data[0, :], self.data[1, :])
        # generate the meshgrid
        X, Y = np.meshgrid(np.linspace(self.range[0, 0], self.range[0, 1], 200),
                           np.linspace(self.range[1, 0], self.range[1, 1], 200))
        # plot the estimated gaussian distribution
        pos = np.stack((X, Y), axis=2)
        # np.einsum reference link:
        # http://ajcr.net/Basic-guide-to-einsum/
        value = np.einsum('...k,kl,...l->...', pos - self.mu.T[0], np.linalg.inv(self.sigma), pos - self.mu.T[0])
        Z = np.exp(- value / 2) / (np.sqrt(np.linalg.det(self.sigma) * (2 * np.pi) ** self.n))
        plt.contour(X, Y, Z)
        plt.show()


if __name__ == "__main__":
    # Random sample the data
    d = np.random.multivariate_normal(.9 * np.ones(2), np.array([[.1, 0], [0, 12]]), 10).T
    # Fit and plot the model
    multigaus = MultiGaussian(d)
    multigaus.fit()
    multigaus.plot2D()

