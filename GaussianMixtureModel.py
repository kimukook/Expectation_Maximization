'''
This is a simple script that fits a Gaussian Mixture Model to the observed data.
The Gaussian mixture model is solved via Expectation Maximization algorithm.
This code follows the psesudo code from Andrew Ng at: http://cs229.stanford.edu/notes/cs229-notes7b.pdf
=========================
Author  :  Muhan Zhao

Date    :  Aug. 17, 2019

Location:  West Hill, LA, CA
=========================
'''

import numpy as np
import matplotlib.pyplot as plt


class GMM:
    def __init__(self, data, num_class):
        self.data = data
        self.n = data.shape[0]
        self.m = data.shape[1]
        self.K = num_class

        self.mu = np.zeros((self.n, self.K))
        self.sigma = np.empty(shape=[self.n, self.n, self.K])
        for i in range(self.K):
            self.sigma[:, :, i] = np.identity(self.n)

        # classify each data point to the j-th distribution
        # The summation of self.w along axis=1 should be 1 for each row
        self.w = np.ones((self.m, self.K)) / self.K
        # Randomly initialize phi, denoting the latent variables

        # The summation of phi should be equal to 1
        self.phi = np.random.rand(self.K)
        self.phi /= np.sum(self.phi)

        self.iter = 0
        self.max_iter = self.m * self.n * 5

        # calculate the plot range
        self.range = np.zeros((self.n, 2))
        # the first column is the minimum - 0.1 * range
        # the second column is the maximum + 0.1 * range
        for i in range(self.n):
            coordinate_range = np.max(self.data[i, :]) - np.min(self.data[i, :])
            self.range[i, 0] = np.min(self.data[i, :]) - .1 * coordinate_range
            self.range[i, 1] = np.max(self.data[i, :]) + .1 * coordinate_range

    def pdf_eval(self, x, mu, cov):
        '''
        Given the point x and Gaussian distribution parameter mu and cov, determine the probability
        :param x:
        :param mu:
        :param cov:
        :return:
        '''
        x = x.reshape(-1, 1)
        mu = mu.reshape(-1, 1)
        return 1 / np.sqrt(np.linalg.det(cov) * (2 * np.pi) ** self.n) \
               * np.exp(-np.dot((x - mu).T, np.dot(np.linalg.inv(cov), x - mu)))

    def e_step(self):
        # loop over all points
        for i in range(self.m):
            # for each point, loop over all the classes
            class_pdf = np.zeros(self.K)
            for j in range(self.K):
                # update the w_ij coefficient
                class_pdf[j] = self.pdf_eval(self.data[:, i], self.mu[:, j], self.sigma[:, :, j])
            self.w[i, :] = (class_pdf * self.phi) / (np.dot(class_pdf, self.phi))

    def m_step(self):
        # loop over all classes
        self.phi = np.mean(self.w, axis=0)
        self.mu = np.dot(self.data, self.w) / (np.sum(self.w, axis=0))
        for i in range(self.K):
            self.sigma[:, :, i] = np.dot(self.w[:, i] * (self.data - self.mu[:, i].reshape(-1, 1)), (self.data - self.mu[:, i].reshape(-1, 1)).T)
            self.sigma[:, :, i] /= np.sum(self.w[:, i])

    def em_solver(self):
        while 1:
            print('EM algorithm starts...')
            old_phi = np.copy(self.phi)
            self.e_step()
            self.m_step()
            self.iter += 1
            # if np.linalg.norm(old_phi - self.phi) < 1e-3:
            #     print('EM converges')
            #     break
            if self.iter == self.max_iter:
                print('EM maximum iteration achieves')
                break

    def plot2D(self):
        assert self.data.shape[0] == 2, 'This is a 2D data visualizer'
        plt.figure()
        # scatter plot the data
        plt.scatter(self.data[0, :], self.data[1, :])
        # generate the meshgrid
        X, Y = np.meshgrid(np.linspace(self.range[0, 0], self.range[0, 1], 200),
                           np.linspace(self.range[1, 0], self.range[1, 1], 200))
        pos = np.stack((X, Y), axis=2)
        # np.einsum reference link:
        # http://ajcr.net/Basic-guide-to-einsum/
        for i in range(self.K):
            value = np.einsum('...k,kl,...l->...', pos - self.mu[:, i], np.linalg.inv(self.sigma[:, :, i]), pos - self.mu[:, i])
            Z = np.exp(- value / 2) / (np.sqrt(np.linalg.det(self.sigma[:, :, i]) * (2 * np.pi) ** self.n))
            plt.contour(X, Y, Z)
        plt.show()


if __name__ == "__main__":
    d1 = np.random.multivariate_normal(.9 * np.ones(2), np.array([[.1, 0], [0, 12]]), 10).T
    d2 = np.random.multivariate_normal(.2 * np.ones(2), np.array([[12, 0], [0, .1]]), 10).T
    d3 = np.random.multivariate_normal(.5 * np.ones(2), np.array([[1, 0], [0, 1]]), 10).T
    d = np.hstack((d1, d2, d3))

    em = GMM(d, 3)
    em.em_solver()
    em.plot2D()
