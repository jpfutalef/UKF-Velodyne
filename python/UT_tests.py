from python.ukf import *
import numpy as np


# Testing functions
def F1D1D(x, u):
    return np.vstack(x / (1 + x ** 2) + 0.5 * u)


def F2D2D(x, u):
    return np.vstack([[x[0, 0] ** 2], [x[1, 0] ** 2]])


def F3D3D(x, u):
    return np.vstack([[x[0] ** 2], [x[1] ** 2], [x[2] ** 2]])


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

# Unscented transformation with previous points test

# UT test scalar
xk = 1.5
uk = 1.0
Pk = 2.0

XK, WK = unscentedTransform(xk, Pk)

print("Sigma points (scalar): ", XK)
print("Weight (scalar): ", WK)

XK1 = evalSigmaPointsWithInput(XK, uk, F1D1D)
xk1 = weightedSum(XK1, WK)

Qk = 1.5
XK1_new, WK_new = unscentedTransform_addPoints(xk1, Qk, XK1)

YK = evalSigmaPoints(XK1_new, H1D1D)
yk = weightedSum(YK, WK_new)

print("Projected sigma points: ", YK)
print("Projected mean: ", yk)
print()

# UT test 2D
xk = np.vstack([[1.5], [2.0]])
uk = 1.0
Pk = 2 * np.eye(2)

XK, WK = unscentedTransform(xk, Pk)

print("Sigma points (2D case): ", XK)
print("Weight (2D case): ", WK)

XK1 = evalSigmaPointsWithInput(XK, uk, F2D2D)
xk1 = weightedSum(XK1, WK)

Qk = 1.5 * np.eye(2)
XK1_new, WK_new = unscentedTransform_addPoints(xk1, Qk, XK1)

YK = evalSigmaPoints(XK1_new, H2D2D)
yk = weightedSum(YK, WK_new)

print("Projected sigma points: ", YK)
print("Projected mean: ", yk)
print()

# UT test 3D
xk = np.array([[1.5], [2.0], [1.8]])
Pk = 2 * np.eye(3)

XK, WK = unscentedTransform(xk, Pk)

print("Sigma points (3D case): ", XK)
print("Weight (3D case): ", WK)

XK1 = evalSigmaPointsWithInput(XK, uk, F3D3D)
xk1 = weightedSum(XK1, WK)

Qk = 1.5 * np.eye(3)
XK1_new, WK_new = unscentedTransform_addPoints(xk1, Qk, XK1)

YK = evalSigmaPoints(XK1_new, H3D3D)
yk = weightedSum(YK, WK_new)

print("Projected sigma points: ", YK)
print("Projected mean: ", yk)
print()
