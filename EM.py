

import numpy as np
from scipy import stats
import pickle
import sys

class GMM(object):

    def __init__(self, X, K):
        X = np.asarray(X)
        self.N, self.d = X.shape
        self.data = X.copy()
        self.K = K

        self.mu_arr = np.random.random((self.K, self.d))
        self.Lambda_arr = np.tile(np.eye(self.d).reshape(1, self.d, self.d), reps=(self.K, 1, 1))
        self.pi = np.ones(self.K) / self.K
        self.gamma = np.empty((self.N, self.K))

    def fit(self, tol=1e-5):
        num_iters = 0
        ll = 1
        previous_ll = 0
        while(ll - previous_ll > tol):
            previous_ll = self.loglikelihood()
            #self.em()
            self.e_step()
            self.m_step()
            num_iters += 1
            ll = self.loglikelihood()
            print('Iteration %d: log-likelihood is %.8f' %(num_iters, ll))

        print('\tTerminate at %d-th iteration: log-likelihood is %.8f' %(num_iters, ll))

        #self.save_params('params.dat')
        #self.save_posterior('z.csv')

    def loglikelihood(self):
        lh = np.empty_like(self.gamma)
        for k in range(self.K):
            gauss_k = stats.multivariate_normal(self.mu_arr[k], self.Lambda_arr[k])
            lh[:, k] = gauss_k.pdf(self.data) * self.pi[k]

        ll = np.log(np.sum(lh, axis=1)).mean()

        return ll

    #def em(self):
    #    self.e_step()
    #    self.m_step()

    def e_step(self):
        for k in range(self.K):
            gauss_k = stats.multivariate_normal(self.mu_arr[k], self.Lambda_arr[k])
            self.gamma[:, k] = gauss_k.pdf(self.data) * self.pi[k]

        self.gamma /= np.sum(self.gamma, axis=1, keepdims=True) # normalize

    def m_step(self):
        # compute sufficient statistics
        S1 = np.sum(self.gamma)
        S_1 = np.sum(self.gamma, axis=0) # size K

        S_x = self.gamma[:, :, np.newaxis] * self.data[:, np.newaxis, :] # size N x K x d
        S_x = np.sum(S_x, axis=0) # size K x d

        xxT = self.data[:,:, np.newaxis] * self.data[:, np.newaxis, :] # size N x d x d
        S_xxT = self.gamma[:, :, np.newaxis, np.newaxis] * xxT[:, np.newaxis, :, :]  # size N x K x d x d
        S_xxT = np.sum(S_xxT, axis=0) # size K x d x d

        # update pi, mu, Lambda
        self.pi = S_1 / S1          # size K
        self.mu_arr = S_x / S_1.reshape(-1, 1)     # size K x d
        #print('lala: ', (S_xxT / S_1.reshape(-1, 1, 1)).shape)
        self.Lambda_arr = S_xxT / S_1.reshape(-1, 1, 1) - self.mu_arr[:, :, np.newaxis] * self.mu_arr[:, np.newaxis, :]   # size K x d x d

    def save_params(self, params_file):
        with open(params_file, 'wb') as file:
            params_dict = {
                'n_components': self.K, # number of clusters
                'weights': self.pi,
                'means': self.mu_arr,
                'covariances': self.Lambda_arr,
            }
            pickle.dump(params_dict, file)

    def save_posterior(self, posterior_file):
        np.savetxt(fname=posterior_file, fmt='%.8f', X=self.gamma, delimiter=', ')



if __name__ == '__main__':

    #X = np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 0.8]], 20)
    #X = np.vstack((X, np.random.multivariate_normal([20, 10], np.identity(2), 50)))
    #=> same results with sk_leran.mixture.GaussianMixture()
    # but much more slower() = use too many for loops in e_step(), loglikelihood()

    try:
        n_components = int(sys.argv[1])
    except:
        n_components = 5 # default

    input_file = sys.argv[-3]
    output_file = sys.argv[-2]
    out_params_file = sys.argv[-1]

    #X = np.loadtxt('x.csv', delimiter=',')
    X = np.loadtxt(input_file, delimiter=',')
    print('Data shape: %s' %(X.shape,))
    print('First data points: \n', X[:2])

    gmm = GMM(X, K=n_components)
    #gmm = GMM(X, K=2)
    gmm.fit()

    gmm.save_params(out_params_file)
    gmm.save_posterior(output_file)

    print('weights: ', gmm.pi)
