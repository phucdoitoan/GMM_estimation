

# For the implementation of VB algorithm, I refer to the following sources:
# https://github.com/JulienNonin/variational-gaussian-mixture
# and Pattern Recognition and Machine Learning (Bishop) chapter 10
# the symbols used in this implementation are following Bishop for convenience.

import numpy as np
from scipy.special import digamma, gammaln, logsumexp
import sys
import pickle

def log_wishart_B(invW, nu):
    D = len(invW)
    return + 0.5 * nu * np.log(np.linalg.det(invW)) \
           - 0.5 * nu * D * np.log(2) \
           - 0.25 * D * (D-1) * np.log(np.pi) \
           - gammaln(0.5 * (nu - np.arange(D))).sum()

def log_dirichlet_C(alpha):
    return gammaln(np.sum(alpha)) - gammaln(alpha).sum()

class VB(object):

    def __init__(self, X, K):

        self.X = X
        self.N, self.d = X.shape
        self.data = X.copy()
        self.K = K

        self.alpha0 = 1. / self.K
        self.beta0 = 1.
        self.m0 = np.zeros(self.d)
        self.nu0 = self.d
        self.invW0 = np.cov(self.X.T)

        self.resp = np.random.rand(self.N, self.K)
        self.resp /= self.resp.sum(axis=1, keepdims=True)  # gamma_n_k as in the slides
        self.m_step(np.log(self.resp))  # run m_step to initiate alpha, beta, m, nu, invW

    def fit(self, max_iter=200, tol=1e-8):

        self.elbo_arr = []
        for i in range(max_iter):
            ln_resp, ln_lambda_tilde, ln_pi_tilde = self.e_step()
            self.m_step(ln_resp)
            elbo = self.elbo(ln_resp, ln_lambda_tilde, ln_pi_tilde)
            self.elbo_arr.append(elbo)
            print('Iteration %s: Evidence Lower Bound is %.8f' %(i, elbo))

            if i >= 1 and (self.elbo_arr[i] - self.elbo_arr[i-1] < tol):
                print('\tTerminate at %s-th iteration: Evidence Lower Bound is %.8f' %(i, elbo))
                break
            elif i == max_iter - 1:
                print('\tTerminate at %s-th iteration (Maximum = %s iter): Evidence Lower Bound is %.8f' % (i, max_iter, elbo))


        self.weights = self.alpha / np.sum(self.alpha)
        self.covs = self.invW / self.nu[:, np.newaxis, np.newaxis]

        # Final e-step to guarantee that the labels are consistent
        ln_resp, *_ = self.e_step()
        self.resp = np.exp(ln_resp)

    def e_step(self):
        W = np.linalg.inv(self.invW)

        E = np.zeros((self.N, self.K)) # Expectation in (10.64)
        for k in range(self.K):
            Xc = X - self.m[k]
            E[:, k] = self.d / self.beta[k] + self.nu[k] * np.sum(Xc @ W[k] * Xc, axis=1)  # (10.64)
        ln_lambda_tilde = np.sum(digamma(0.5 * (self.nu - np.arange(0, self.d)[:, np.newaxis])), axis=0) \
                          + self.d * np.log(2) + np.log(np.linalg.det(W))  # (10.65)
        ln_pi_tilde = digamma(self.alpha) - digamma(np.sum(self.alpha))  # (10.66)

        ln_rho = ln_pi_tilde + 0.5 * ln_lambda_tilde - 0.5 * (E + self.d * np.log(2 * np.pi))  # (10.46)
        ln_resp = ln_rho - np.c_[logsumexp(ln_rho, axis=1)]  # (10.49)

        # output ln_resp, ln_lambda_tilde and ln_pi_tilde in order to compute elbo
        return ln_resp, ln_lambda_tilde, ln_pi_tilde

    def m_step(self, ln_resp):
        # update in m_step
        N, x_bar, S = self.compute_statististics(np.exp(ln_resp))

        self.alpha = self.alpha0 + N  # (10.58)
        self.beta = self.beta0 + N  # (10.60)
        self.m = (self.beta0 * self.m0 + x_bar * N[:, np.newaxis]) / self.beta[:, np.newaxis]  # (10.61) means
        self.nu = self.nu0 + N  # (10.63)

        self.invW = np.zeros((self.K, self.d, self.d))  # covariances
        for k in range(self.K):
            xc = x_bar[k] - self.m0
            self.invW[k] = self.invW0 + N[k] * S[k] + (self.beta0 * N[k]) * (xc @ xc.T) / self.beta[k]  # (10.62)

    def compute_statististics(self, resp):
        # compute sufficient statistics as in 10.51 - 10.53 of bishop
        # i.e compute S_1, S_x, S_xxT as in the slides
        N = resp.sum(axis=0)  # (10.51)
        x_bar = (resp.T @ X) / N[:, np.newaxis]  # (10.52)
        S = np.zeros((self.K, self.d, self.d))
        for k in range(self.K):
            Xc = X - x_bar[k]
            S[k] = ((resp[:, k] * Xc.T) @ Xc) / N[k]  # (10.53)

        return N, x_bar, S

    def elbo(self, ln_resp, ln_Lambda_tilde, ln_pi_tilde):
        # compute elbo according to Bishop page 481-482
        resp = np.exp(ln_resp)

        # compute the sufficient statistics S_1, S_k, S_xxT in the forms stated in Bishop
        # for easier implementation using formulas in page 481-482 of Bishop
        N, x_bar, S = self.compute_statististics(resp)

        W = np.linalg.inv(self.invW)

        ln_p_x = 0.5 * np.sum(N * ln_Lambda_tilde) \
                 - 0.5 * self.d * np.sum(N / self.beta) \
                 - 0.5 * np.sum(N * self.nu * np.trace(S @ W, axis1=1, axis2=2)) \
                 - 0.5 * np.sum([N[k] * self.nu[k] * (x_bar[k] - self.m[k]) @ W[k] @ (x_bar[k] - self.m[k]) for k in range(self.K)]) \
                 - 0.5 * N.sum() * self.d * np.log(2 * np.pi)  # (10.71)
        ln_p_z = np.sum(resp * ln_pi_tilde)  # (10.72)
        ln_p_pi = (self.alpha0 - 1) * ln_pi_tilde.sum() + log_dirichlet_C([self.alpha0] * self.K)  # (10.73)
        ln_p_mu_lambda = 0.5 * np.sum(ln_Lambda_tilde) \
                         + 0.5 * self.K * self.d * np.log(0.5 * self.beta0 / np.pi) \
                         - 0.5 * self.d * (self.beta0 / self.beta).sum() \
                         - 0.5 * self.beta0 * np.sum([self.nu[k] * (self.m[k] - self.m0) @ W[k] @ (self.m[k] - self.m0) for k in range(self.K)]) \
                         + self.K * log_wishart_B(self.invW0, self.nu0) \
                         + 0.5 * (self.nu0 - self.d - 1) * ln_Lambda_tilde.sum() \
                         - 0.5 * np.sum(self.nu * np.trace(self.invW0 @ W, axis1=1, axis2=2))  # (10.74)

        ln_q_z = np.sum(resp * ln_resp)  # (10.75)
        ln_q_pi = np.sum((self.alpha - 1) * ln_pi_tilde) + log_dirichlet_C(self.alpha)  # (10.76)
        ln_q_mu_lambda = 0.5 * np.sum(ln_Lambda_tilde) - 0.5 * self.K * self.d \
                         + np.sum(0.5 * self.d * np.log(0.5 * self.beta / np.pi)) \
                         + np.sum([log_wishart_B(self.invW[k], self.nu[k]) for k in range(self.K)]) \
                         + np.sum(0.5 * (self.nu - self.d - 1) * ln_Lambda_tilde) \
                         - np.sum(0.5 * self.nu * self.d)  # (10.77)

        return (ln_p_x + ln_p_z + ln_p_pi + ln_p_mu_lambda - ln_q_z - ln_q_pi - ln_q_mu_lambda) / self.N  # mean of elbo

    def save_params(self, params_file):
        with open(params_file, 'wb') as file:
            params_dict = {
                'n_components': self.K, # number of clusters
                'weights': self.weights,
                'alpha': self.alpha,
                'beta': self.beta,
                'm': self.m,
                'nu': self.nu,
                'invW': self.invW
            }
            pickle.dump(params_dict, file)

    def save_posterior(self, posterior_file):
        np.savetxt(fname=posterior_file, fmt='%.8f', X=self.resp, delimiter=', ')

if __name__ == '__main__':
    #X = np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 0.8]], 20)
    #X = np.vstack((X, np.random.multivariate_normal([20, 10], np.identity(2), 50)))
    #vbgmm = VB(X, 2)
    #vbgmm.fit()

    try:
        n_components = int(sys.argv[1])
    except:
        n_components = 5  # default

    input_file = sys.argv[-3]
    output_file = sys.argv[-2]
    out_params_file = sys.argv[-1]

    # X = np.loadtxt('x.csv', delimiter=',')
    X = np.loadtxt(input_file, delimiter=',')
    print('Data shape: %s' % (X.shape,))
    print('First data points: \n', X[:2])

    vbgmm = VB(X, K=n_components)
    vbgmm.fit()

    vbgmm.save_params(out_params_file)
    vbgmm.save_posterior(output_file)

    print('weights: ', vbgmm.weights)
