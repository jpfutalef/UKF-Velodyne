from python.ukf import *
import numpy as np
import matplotlib.pyplot as plt


# Testing functions
def F1D1D(x, u):
    return np.vstack([x[0, 0] / (1 + x[0, 0] ** 2) + 0.5 * u])


def F2D2D(x, u):
    return np.vstack([[x[0, 0] ** 2], [x[1, 0] ** 2]])


def F2D2D_linearModel(x, u):
    return np.vstack([x[0, 0] + u[0, 0], x[1, 0] + u[1, 0]])


def F3D3D(x, u):
    return np.vstack([[x[0] ** 2], [x[1] ** 2], [x[2] ** 2]])


def H1D1D(x):
    return np.vstack([x[0, 0] ** 2])


def H2D2D(x):
    return np.vstack([[x[0, 0] ** 2], [x[1, 0] ** 2]])


def H2D2D_cartesianToPolar(x):
    r = np.sqrt(x[0, 0] ** 2 + x[1, 0] ** 2)
    theta = np.arctan2(x[1, 0], x[0, 0])  # ensures the range (-pi, pi] for quadrants I, II and IV
    if x[0, 0] < 0 and x[1, 0] < 0:
        theta += 2 * np.pi  # ensures the range (0, 2pi] for quadrant III
    return np.vstack([r, theta])


def H3D3D(x):
    return np.vstack([[x[0] ** 2], [x[1] ** 2], [x[2] ** 2]])


# Unscented Kalman filter tests

# Time to simulate
tEnd = 10
t = np.arange(tEnd + 1)

# UKF test 2D

xk0 = np.vstack([2.0, 2.0])
PK0 = np.eye(2)
uk = np.vstack([1.0, 1.0])

Q = 0.2 * np.eye(2)
R = 0.5 * np.eye(2)

yk = H2D2D_cartesianToPolar(xk0)

x = np.empty([np.size(xk0), tEnd + 1])
y = np.empty([np.size(yk), tEnd + 1])
xEst = np.empty([np.size(xk0), tEnd + 1])

y[:, 0] = yk[:, 0]
x[:, 0] = xk0[:, 0]
xEst[:, 0] = xk0[:, 0]

for k in t[1:]:
    xk1 = F2D2D_linearModel(xk0, uk) + np.vstack(np.random.normal(scale=[Q[0, 0], Q[1, 1]]))
    yk = H2D2D_cartesianToPolar(xk1) + np.vstack(np.random.normal(scale=[R[0, 0], R[1, 1]]))
    x[:, k] = xk1[:, 0]
    y[:, k] = yk[:, 0]
    xk0, PK0 = UKF_additiveNoise(xk0, PK0, uk, yk, Q, R, F2D2D_linearModel, H2D2D_cartesianToPolar)
    xEst[:, k] = xk0[:, 0]

plt.plot(x[0, :], x[1, :])
plt.plot(xEst[0, :], xEst[1,:])
plt.show()
