from python.ukf import *
import numpy as np


# Testing functions
def F1D1D(x, u):
    return np.vstack(x / (1 + x ** 2) + 0.5 * u)


def H1D1D(x):
    return np.vstack(x ** 2)


def H2D2D(x):
    return np.vstack([[x[0, 0] ** 2], [x[1, 0] ** 2]])


def H3D3D(x):
    return np.vstack([[x[0] ** 2], [x[1] ** 2], [x[2] ** 2]])


# Unscented transformation tests

# UT test scalar
xk = 1.5
Pk = 2.0

XK, WK = unscentedTransform(xk, Pk)

print("Sigma points (scalar): ", XK)
print("Weight (scalar): ", WK)

YK = evalSigmaPoints(XK, H1D1D)
yk = weightedSum(YK, WK)

print("Projected sigma points: ", YK)
print("Projected mean: ", yk)
print()

# UT test 2D
xk = np.vstack([[1.5], [2.0]])
Pk = 2 * np.eye(2)

XK, WK = unscentedTransform(xk, Pk)

print("Sigma points (2D case): ", XK)
print("Weight (2D case): ", WK)

YK = evalSigmaPoints(XK, H2D2D)
yk = weightedSum(YK, WK)

print("Projected sigma points: ", YK)
print("Projected mean: ", yk)
print()

# UT test 3D
xk = np.array([[1.5], [2.0], [1.8]])
Pk = 2 * np.eye(3)

XK, WK = unscentedTransform(xk, Pk)

print("Sigma points (3D case): ", XK)
print("Weight (3D case): ", WK)

YK = evalSigmaPoints(XK, H3D3D)
yk = weightedSum(YK, WK)



print("Projected sigma points: ", YK)
print("Projected mean: ", yk)
print()
