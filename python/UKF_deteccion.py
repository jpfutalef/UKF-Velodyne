from python.ukf import *
# from python.personDetectionVelodyne import *
import numpy as np
import matplotlib.pyplot as plt
# import rosbag
import pandas as pd


# Testing functions
def F2D2D_linearModel(x, u):
    return np.vstack([x[0, 0] + u[0, 0],
                      x[1, 0] + u[1, 0]])


def H2D2D_cartesianToPolar(x):
    r = np.sqrt(x[0, 0] ** 2 + x[1, 0] ** 2)
    theta = np.arctan2(x[1, 0], x[0, 0])  # ensures the range (-pi, pi] for quadrants I, II and IV
    if x[0, 0] < 0 and x[1, 0] < 0:
        theta += 2 * np.pi  # ensures the range (0, 2pi] for quadrant III
    return np.vstack([r,
                      theta])


def F4D4D_linearModel(x, u):
    # x_k+1 = x_k + vx*dt_k y lo mismo para y_k+1. u[0,0] = dt_k
    return np.vstack([x[0, 0] + x[1, 0] * u[0, 0],
                      x[1, 0],
                      x[2, 0] + x[3, 0] * u[0, 0],
                      x[3, 0]])


def H4D2D_cartesianToPolar(x):
    r = np.sqrt(x[0, 0] ** 2 + x[2, 0] ** 2)
    theta = np.arctan2(x[2, 0], x[0, 0])  # ensures the range (-pi, pi] for quadrants I, II and IV
    if x[0, 0] < 0 and x[2, 0] < 0:
        theta += 2 * np.pi  # ensures the range (0, 2pi] for quadrant III
    return np.vstack([r,
                      theta])


def H4D4D_identity(x):
    return np.vstack([x[0, 0],
                      x[1, 0],
                      x[2, 0],
                      x[3, 0]])


'''
# para formar clusters raw
bagfile = rosbag.Bag('../data/2.bag', 'r')
clustersToCSV(bagfile, "bag2")

'''

# UKF para detecciones ya obtenidas

vals = pd.read_csv('matlab/end_clusters_bag2.csv').values

# condiciones iniciales
xk0 = np.vstack([0.0,
                 vals[0, 2] / vals[0, 0],
                 0.0,
                 vals[0, 3] / vals[0, 0]])  # estado inicial propuesto
PK0 = 1.5*np.eye(4)  # matriz de covarianza inicial
t0 = 0.0  # tiempo inicial

# parametros
sigma2_x = 0.01  # ruido proceso x
sigma2_y = 0.01  # ruido proceso y
sigma2_vx = 0.01  # ruido proceso y
sigma2_vy = 0.01  # ruido proceso y

range_accuracy = 0.03  # cm
angular_resolution = 0.4  # deg

for k in range(0, np.size(vals, 0)):
    t1 = vals[k, 0]
    dt = t1 - t0
    uk = np.vstack([dt])

    # covarianza proceso
    Q = np.array([[sigma2_x, 0, 0, 0],
                  [0, sigma2_vx, 0, 0],
                  [0, 0, sigma2_y, 0],
                  [0, 0, 0, sigma2_vy]])
    # covarianza medicion
    R = np.array([[np.power(range_accuracy, 2), 0.0],
                  [0.0, np.power(np.deg2rad(angular_resolution), 2)]])

    yk = H4D2D_cartesianToPolar(np.vstack([vals[k, 2], 0.0, vals[k, 3], 0.0]))  # obtener medicion
    xk0, PK0 = UKF_additiveNoise(xk0, PK0, uk, yk, Q, R, F4D4D_linearModel, H4D2D_cartesianToPolar)

    '''
    
    # covarianza proceso
    Q = 0.01*np.eye(4)
    # covarianza medicion
    R = np.array([[sigma2_x, 0, 0, 0],
                  [0, 0.5, 0, 0],
                  [0, 0, sigma2_x, 0],
                  [0, 0, 0, 0.5]])
    yk = H4D4D_identity(np.vstack([vals[k, 2], 0.0, vals[k, 3], 0.0]))
    xk0, PK0 = UKF_additiveNoise(xk0, PK0, uk, yk, Q, R, F4D4D_linearModel, H4D4D_identity)
    '''
    print("Measured output:\n", yk)
    print("Output with estimated state: \n", H4D2D_cartesianToPolar(xk0))
    print("Estimated state:\n", xk0)
    print("Q: \n", Q)
    print("R: \n", R)
    print("Covariance P0k:\n", PK0, "\n")
    t0 = t1

plt.show()
