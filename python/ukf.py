import numpy as np
from scipy.linalg import sqrtm as msqrt


def unscentedTransform(x, P, kappa=1.0):
    n = np.size(x)
    a = np.sqrt(n + kappa)

    # non-scalar case
    if np.size(P) > 1:
        sP = msqrt(P)
        X = np.empty([n, 2 * n + 1])
        W = np.empty([1, 2 * n + 1])
        X[:, 0] = x[:, 0]
        W[0, 0] = kappa / (n + kappa)
        for i in range(0, n):
            X[:, i + 1] = x[:, 0] + a * sP[:, i]
            X[:, i + n + 1] = x[:, 0] - a * sP[:, i]
            W[:, i + 1] = 1 / (2.0 * (n + kappa))
            W[:, i + n + 1] = 1 / (2.0 * (n + kappa))

    # scalar case
    else:
        sP = np.sqrt(P)
        X = np.empty([1, 3])
        W = np.empty([1, 3])

        X[0, 0] = x
        X[0, 1] = x + a * sP
        X[0, 2] = x - a * sP

        W[0, 0] = kappa / (n + kappa)
        W[0, 1] = 1 / (2.0 * (n + kappa))
        W[0, 2] = 1 / (2.0 * (n + kappa))

    return X, W


def sigmaPoints_addPoints(x, Q, Xprev, kappa=1.0):
    n = np.size(x, 0)
    a = np.sqrt(2.0 * n + kappa)
    '''
    sP = msqrt(Q)

    X = np.empty([n, 4 * n + 1])
    W = np.empty([1, 4 * n + 1])
    X[:, 0:2 * n + 1] = Xprev
    W[0,0] = kappa / (2.0 * n + kappa)
    for i in range(0, n):
        X[:, i + 2 * n + 1] = x + a * sP[:,i - 1]
        X[:, i + 3 * n + 1] = x - a * sP[:,i - 1]
        W[0, i + 1] = 1 / (2.0 * (2.0 * n + kappa))
        W[0, i + n + 1] = 1 / (2.0 * (2.0 * n + kappa))
        W[0, i + 2*n + 1] = 1 / (2.0 * (2.0 * n + kappa))
        W[0, i + 3*n + 1] = 1 / (2.0 * (2.0 * n + kappa))
    '''

    if np.size(Q) > 1:
        sP = msqrt(Q)

        X = np.empty([n, 4 * n + 1])
        W = np.empty([1, 4 * n + 1])
        X[:, 0:2 * n + 1] = Xprev
        W[0, 0] = kappa / (2.0 * n + kappa)

        for i in range(0, n):
            X[:, i + 2 * n + 1] = x + a * sP[:, i - 1]
            X[:, i + 3 * n + 1] = x - a * sP[:, i - 1]
            W[0, i + 1] = 1 / (2.0 * (2.0 * n + kappa))
            W[0, i + n + 1] = 1 / (2.0 * (2.0 * n + kappa))
            W[0, i + 2 * n + 1] = 1 / (2.0 * (2.0 * n + kappa))
            W[0, i + 3 * n + 1] = 1 / (2.0 * (2.0 * n + kappa))
    else:
        sP = np.sqrt(Q)
        X = np.empty([1, 5])
        W = np.empty([1, 5])

        X[0, 0:3] = Xprev
        X[0, 3] = x + a * sP
        X[0, 4] = x - a * sP

        W[0, 0] = kappa / (2.0 * n + kappa)
        W[0, 1:5] = np.ones([1, 4]) / (2.0 * (2.0 * n + kappa))

    return X, W


def weightedSum(X, W):
    return np.average(X, axis=1, weights=W[0, :])


def evalSigmaPoints(sp, f):
    firstEvaluatedSP = f(np.vstack(sp[:, 0]))
    nColumns = np.size(sp, 1)
    nRows = np.size(firstEvaluatedSP, 0)

    evaluatedSP = np.empty([nRows, nColumns])

    evaluatedSP[:, 0] = firstEvaluatedSP[:, 0]
    for i in range(1, nColumns):
        col = f(np.vstack(sp[:, i]))
        evaluatedSP[:, i] = col[:, 0]
    return evaluatedSP


def UKF_additiveNoise(xk0, Pk0, uk0, yk1, Q, R, F, H, kappa=1):
    Xk0, Wk0 = unscentedTransform(xk0, Pk0)

    # Prediccion
    Xk1_prior = np.empty_like(Xk0)
    for i in range(0, np.size(Xk0, 1)):
        Xk1_prior[:, i] = F(Xk0[:, i], uk0)

    xk1_prior = weightedSum(Xk1_prior, Wk0)
    Pk1_prior = Q
    for i in range(0, np.size(Xk0, 1)):
        v = Xk1_prior[:, i] - xk1_prior
        Pk1_prior += Wk0[0, i] * np.outer(v, v)

    # Propagar prediccion
    Xk1_prior_new, Wk1 = unscentedTransform(xk1_prior, Q)
    Yk1_prior = np.empty([np.size(yk1), np.size(Xk1_prior_new, 1)])
    for i in range(0, np.size(Yk1_prior, 1)):
        Yk1_prior[:, i] = H(Xk1_prior_new[:, i])

    yk1_prior = weightedSum(Yk1_prior, Wk1)

    # Correccion
    Pyy = R
    for i in range(0, np.size(Yk1_prior, 1)):
        v = Yk1_prior[:, i] - yk1_prior
        Pyy += Wk1[0, i] * np.outer(v, v)

    Pxy = np.empty([np.size(xk0), np.size(yk1)])
    for i in range(0, np.size(Yk1_prior, 1)):
        v = Xk1_prior_new[:, i] - xk1_prior
        w = Yk1_prior[:, i] - yk1_prior
        Pxy += Wk1[0, i] * np.outer(v, w)

    K = np.matmul(Pxy, np.linalg.inv(Pyy))
    xk1 = xk1_prior + np.matmul(K, yk1 - yk1_prior)
    Pk1 = Pk1_prior - np.matmul(K, np.outer(Pyy, K))
    return xk1, Pk1
