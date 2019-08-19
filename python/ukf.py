import numpy as np
from scipy.linalg import sqrtm as msqrt


def unscentedTransform(x, P, kappa=0.0, alpha=1e-3, beta=2.0, algo='Julier'):
    L = np.size(x)

    # non-scalar case using Wan's method to calculate sigma points weights
    if L > 1 and algo == 'Wan':
        lambda_ = np.power(alpha, 2) * (L + kappa) - L
        a = np.sqrt(L + lambda_)

        sP = msqrt(P)

        X = np.zeros([L, 2 * L + 1])
        Wm = np.zeros([1, 2 * L + 1])
        Wc = np.zeros([1, 2 * L + 1])

        X[:, 0] = x[:, 0]
        Wm[0, 0] = lambda_ / (L + lambda_)
        Wc[0, 0] = lambda_ / (L + lambda_) + 1.0 - alpha ** 2 + beta

        for i in range(0, L):
            X[:, i + 1] = x[:, 0] + a * sP[:, i]
            X[:, i + L + 1] = x[:, 0] - a * sP[:, i]
            Wm[:, i + 1] = 1 / (2.0 * (L + lambda_))
            Wc[:, i + 1] = 1 / (2.0 * (L + lambda_))
            Wm[:, i + L + 1] = 1 / (2.0 * (L + lambda_))
            Wc[:, i + L + 1] = 1 / (2.0 * (L + lambda_))

        return X, Wm, Wc

    # non-scalar case using Julier's method to calculate sigma points weights
    elif L > 1 and algo == 'Julier':
        a = np.sqrt(L + kappa)

        sP = msqrt(P)

        X = np.zeros([L, 2 * L + 1])
        W = np.zeros([1, 2 * L + 1])

        X[:, 0] = x[:, 0]
        W[0, 0] = kappa / (L + kappa)

        for i in range(0, L):
            X[:, i + 1] = x[:, 0] + a * sP[:, i]
            X[:, i + L + 1] = x[:, 0] - a * sP[:, i]
            W[:, i + 1] = 1 / (2.0 * (L + kappa))
            W[:, i + L + 1] = 1 / (2.0 * (L + kappa))

        return X, W

    # scalar case
    else:
        a = np.sqrt(L + kappa)
        sP = np.sqrt(P)
        X = np.zeros([1, 3])
        Wm = np.zeros([1, 3])
        Wc = np.zeros([1, 3])

        X[0, 0] = x
        X[0, 1] = x + a * sP
        X[0, 2] = x - a * sP

        Wm[0, 0] = kappa / (L + kappa)
        Wc[0, 0] = kappa / (L + kappa)
        Wm[0, 1] = 1 / (2.0 * (L + kappa))
        Wc[0, 1] = 1 / (2.0 * (L + kappa))
        Wm[0, 2] = 1 / (2.0 * (L + kappa))
        Wc[0, 2] = 1 / (2.0 * (L + kappa))

        return X, Wm, Wc


def unscentedTransform_addPoints(x, Q, Xprev, kappa=0.0, alpha=1e-3, beta=2.0,
                                 algo='Julier'):
    n = np.size(x)
    L = 2 * n

    # non-scalar case
    if n > 1 and algo == 'Wan':
        lambda_ = np.power(alpha, 2) * (L + kappa) - L
        a = np.sqrt(L + lambda_)

        sP = msqrt(Q)

        X = np.zeros([n, 2 * L + 1])
        Wm = np.zeros([1, 2 * L + 1])
        Wc = np.zeros([1, 2 * L + 1])

        X[:, 0:L + 1] = Xprev
        Wm[0, 0] = lambda_ / (L + lambda_)
        Wc[0, 0] = lambda_ / (L + lambda_) + 1.0 - alpha ** 2 + beta

        for i in range(0, n):
            X[:, i + 2 * n + 1] = x[:, 0] + a * sP[:, i - 1]
            X[:, i + 3 * n + 1] = x[:, 0] - a * sP[:, i - 1]

            Wm[0, i + 1] = 1 / (2.0 * (n + kappa))
            Wc[0, i + 1] = 1 / (2.0 * (n + kappa))
            Wm[0, i + n + 1] = 1 / (2.0 * (n + kappa))
            Wc[0, i + n + 1] = 1 / (2.0 * (n + kappa))

            Wm[0, i + 2 * n + 1] = 1 / (2.0 * (n + kappa))
            Wc[0, i + 2 * n + 1] = 1 / (2.0 * (n + kappa))
            Wm[0, i + 3 * n + 1] = 1 / (2.0 * (n + kappa))
            Wc[0, i + 3 * n + 1] = 1 / (2.0 * (n + kappa))

        return X, Wm, Wc

    # non-scalar case
    elif n > 1 and algo == 'Julier':
        a = np.sqrt(L + kappa)

        sP = msqrt(Q)

        X = np.zeros([n, 2 * L + 1])
        W = np.zeros([1, 2 * L + 1])

        X[:, 0:L + 1] = Xprev
        W[0, 0] = kappa / (L + kappa)

        for i in range(0, n):
            X[:, i + 2 * n + 1] = x[:, 0] + a * sP[:, i - 1]
            X[:, i + 3 * n + 1] = x[:, 0] - a * sP[:, i - 1]

            W[0, i + 1] = 1 / (2.0 * (L + kappa))
            W[0, i + n + 1] = 1 / (2.0 * (L + kappa))
            W[0, i + 2 * n + 1] = 1 / (2.0 * (L + kappa))
            W[0, i + 3 * n + 1] = 1 / (2.0 * (L + kappa))

        return X, W

    # scalar case
    else:
        a = np.sqrt(L + kappa)
        sP = np.sqrt(Q)
        X = np.zeros([1, 5])
        Wm = np.zeros([1, 5])
        Wc = np.zeros([1, 5])

        X[0, 0:3] = Xprev
        X[0, 3] = x + a * sP
        X[0, 4] = x - a * sP

        Wm[0, 0] = kappa / (2.0 * n + kappa)
        Wc[0, 1:5] = np.ones([1, 4]) / (2.0 * (2.0 * n + kappa))

        return X, Wm, Wc


def weightedSum(X, W):
    return np.vstack(np.average(X, axis=1, weights=W[0, :]))


def evalSigmaPoints(sp, f):
    firstEvaluatedSP = f(np.vstack(sp[:, 0]))
    nColumns = np.size(sp, 1)
    nRows = np.size(firstEvaluatedSP, 0)

    evaluatedSP = np.zeros([nRows, nColumns])

    evaluatedSP[:, 0] = firstEvaluatedSP[:, 0]
    for i in range(1, nColumns):
        evaluatedSP[:, i] = f(np.vstack(sp[:, i]))[:, 0]
    return evaluatedSP


def evalSigmaPointsWithInput(sp, u, f):
    firstEvaluatedSP = f(np.vstack(sp[:, 0]), u)
    nColumns = np.size(sp, 1)
    nRows = np.size(firstEvaluatedSP, 0)

    evaluatedSP = np.zeros([nRows, nColumns])

    evaluatedSP[:, 0] = firstEvaluatedSP[:, 0]
    for i in range(1, nColumns):
        evaluatedSP[:, i] = f(np.vstack(sp[:, i]), u)[:, 0]
    return evaluatedSP


def UKF_additiveNoise(xk0, Pk0, uk0, yk1, Q, R, F, H, algo='Julier', kappa=1.0,
                      recycleSigmaPoints=True, alpha=1e-3, beta=2.0, ):
    if algo == 'Wan':
        Xk0, Wmk0, Wck0 = unscentedTransform(xk0, Pk0, algo=algo, kappa=kappa)

        # Prediction
        Xk1_prior = evalSigmaPointsWithInput(Xk0, uk0, F)
        xk1_prior = weightedSum(Xk1_prior, Wmk0)
        Pk1_prior = np.copy(Q)
        for i in range(0, np.size(Xk1_prior, 1)):
            v = np.vstack(Xk1_prior[:, i]) - xk1_prior
            Pk1_prior += Wck0[:, i] * np.outer(v, v)

        # Propagate prediction
        if recycleSigmaPoints:
            Xk1_prior_new, Wmk1, Wck1 = unscentedTransform_addPoints(xk1_prior, Q, Xk1_prior,
                                                                     algo=algo, kappa=kappa,
                                                                     alpha=alpha, beta=beta)
        else:
            Xk1_prior_new, Wk1 = unscentedTransform(xk1_prior, Q, algo=algo, kappa=kappa,
                                                    alpha=alpha, beta=beta)

        Yk1_prior = evalSigmaPoints(Xk1_prior_new, H)
        yk1_prior = weightedSum(Yk1_prior, Wmk1)

        # Correction
        Pyy = np.copy(R)
        for i in range(0, np.size(Yk1_prior, 1)):
            v = np.vstack(Yk1_prior[:, i]) - yk1_prior
            Pyy += Wck1[0, i] * np.outer(v, v)

        Pxy = np.zeros([np.size(xk0), np.size(yk1)])
        for i in range(0, np.size(Yk1_prior, 1)):
            v = np.vstack(Xk1_prior_new[:, i]) - xk1_prior
            w = np.vstack(Yk1_prior[:, i]) - yk1_prior
            Pxy += Wck1[0, i] * np.outer(v, w)

        K = np.matmul(Pxy, np.linalg.inv(Pyy))
        xk1 = xk1_prior + np.matmul(K, yk1 - yk1_prior)
        Pk1 = Pk1_prior - np.matmul(np.matmul(K, Pyy), K.T)
        return xk1, Pk1

    elif algo == 'Julier':
        Xk0, Wk0 = unscentedTransform(xk0, Pk0, algo=algo, kappa=kappa)

        # Prediction
        Xk1_prior = evalSigmaPointsWithInput(Xk0, uk0, F)
        xk1_prior = weightedSum(Xk1_prior, Wk0)
        Pk1_prior = np.copy(Q)
        for i in range(0, np.size(Xk1_prior, 1)):
            v = np.vstack(Xk1_prior[:, i]) - xk1_prior
            Pk1_prior += Wk0[0, i] * np.outer(v, v)

        # Propagate prediction
        if recycleSigmaPoints:
            Xk1_prior_new, Wk1 = unscentedTransform_addPoints(xk1_prior, Q, Xk1_prior,
                                                              algo=algo, kappa=kappa)
        else:
            Xk1_prior_new, Wk1 = unscentedTransform(xk1_prior, Q,
                                                    algo=algo, kappa=kappa)

        Yk1_prior = evalSigmaPoints(Xk1_prior_new, H)
        yk1_prior = weightedSum(Yk1_prior, Wk1)

        # Correction
        Pyy = np.copy(R)
        for i in range(0, np.size(Yk1_prior, 1)):
            v = np.vstack(Yk1_prior[:, i]) - yk1_prior
            Pyy += Wk1[0, i] * np.outer(v, v)

        Pxy = np.zeros([np.size(xk0), np.size(yk1)])
        for i in range(0, np.size(Yk1_prior, 1)):
            v = np.vstack(Xk1_prior_new[:, i]) - xk1_prior
            w = np.vstack(Yk1_prior[:, i]) - yk1_prior
            Pxy += Wk1[0, i] * np.outer(v, w)

        K = np.matmul(Pxy, np.linalg.inv(Pyy))
        xk1 = xk1_prior + np.matmul(K, yk1 - yk1_prior)
        Pk1 = Pk1_prior - np.matmul(np.matmul(K, Pyy), K.T)
        return xk1, Pk1

    return
