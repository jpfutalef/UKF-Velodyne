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
    theta = np.arctan2(x[1, 0], x[0, 0])
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
    theta = np.arctan2(x[2, 0], x[0, 0])
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
PK0 = 1.0 * np.eye(4)  # matriz de covarianza inicial
t0 = 0.0  # tiempo inicial

# parametros
sigma_x = 0.1  # ruido proceso x
sigma_y = 0.2  # ruido proceso y
sigma_vx = 0.1  # ruido proceso y
sigma_vy = 0.2  # ruido proceso y

range_accuracy = 0.03  # m
angular_resolution = 0.4  # deg

# vectores a guardar
t = np.zeros([1, 1])

xEst = np.zeros([1, 1])
yEst = np.zeros([1, 1])
rEst = np.zeros([1, 1])
thetaEst = np.zeros([1, 1])

xMea = np.zeros([1, 1])
yMea = np.zeros([1, 1])
rMea = np.zeros([1, 1])
thetaMea = np.zeros([1, 1])

for k in range(0, np.size(vals, 0)):
    t1 = vals[k, 0]
    dt = t1 - t0

    uk = np.vstack([dt])

    # covarianza proceso
    Q = np.array([[np.power(sigma_x, 2), 0, 0, 0],
                  [0, np.power(sigma_vx, 2), 0, 0],
                  [0, 0, np.power(sigma_y, 2), 0],
                  [0, 0, 0, np.power(sigma_vy, 2)]])

    # covarianza medicion
    R = np.array([[np.power(range_accuracy, 2), 0.0],
                  [0.0, np.power(np.deg2rad(angular_resolution), 2)]])

    # obtener medicion
    yk = H4D2D_cartesianToPolar(np.vstack([vals[k, 2], 0.0, vals[k, 3], 0.0]))

    # estimar estado con UKF
    xk0, PK0 = UKF_additiveNoise(xk0, PK0, uk, yk, Q, R,
                                 F4D4D_linearModel, H4D2D_cartesianToPolar,
                                 algo='Julier', kappa=-1.0)
    '''

    xk0, PK0 = UKF_additiveNoise(xk0, PK0, uk, yk, Q, R,
                                 F4D4D_linearModel, H4D2D_cartesianToPolar,
                                 algo='Wan', kappa=0.0, alpha=1e-3, beta=2.0)

    '''

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

    # guardar en vectores
    xEst = np.append(xEst, xk0[0, 0])
    yEst = np.append(yEst, xk0[2, 0])
    xMea = np.append(xMea, vals[k, 2])
    yMea = np.append(yMea, vals[k, 3])

    ykEst = H4D2D_cartesianToPolar(xk0)
    rEst = np.append(rEst, ykEst[0, 0])
    thetaEst = np.append(thetaEst, ykEst[1, 0])
    rMea = np.append(rMea, yk[0, 0])
    thetaMea = np.append(thetaMea, yk[1, 0])

    t = np.append(t, t1)

    # actualizar tiempo inicial para dt
    t0 = t1

# plotting

plt.subplot(221)
plt.plot(t, xMea)
plt.plot(t, xEst)
plt.legend(('medido', 'estimado'))
plt.title('X')

plt.subplot(222)
plt.plot(t, yMea)
plt.plot(t, yEst)
plt.legend(('medido', 'estimado'))
plt.title('Y')

plt.subplot(223)
plt.plot(t, rMea)
plt.plot(t, rEst)
plt.legend(('medido', 'estimado'))
plt.title('r')

plt.subplot(224)
plt.plot(t, thetaMea)
plt.plot(t, thetaEst)
plt.legend(('medido', 'estimado'))
plt.title('theta')

plt.show()


