from ukf import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from animateClusters import *
import rosbag


# FUNCIONES DE PROCESO Y MEDICION. PUEDEN AGREGARSE O QUITARSE

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


### BAGFILE
bagName = '2'

if len(sys.argv) > 1:
    bagName = sys.argv[1]

csvPath = '../data/labeledClusters/end_clusters_' + bagName + '.csv'
bagPath = '../data/rawData/' + bagName + '.bag'
vals = pd.read_csv(csvPath).values

### PARAMETROS MODIFICABLES

# condiciones iniciales
xk0 = np.vstack([vals[0, 2],
                 vals[0, 2] / vals[0, 0],
                 vals[0, 3],
                 vals[0, 3] / vals[0, 0]])  # estado inicial propuesto
PK0 = 1.0 * np.eye(4)  # matriz de covarianza inicial
t0 = 0.0  # tiempo inicial

# Funciones de proceso y medicion
F = F4D4D_linearModel
H = H4D2D_cartesianToPolar

# Ruido proceso
sigma_vx = .05  # ruido proceso vx  0.05 0.5 1.5
sigma_vy = 1.5  # ruido proceso vy  0.05 0.5 1.5

# Ruido medicion
range_accuracy = 0.03  # m
angular_resolution = .4  # deg

# dispersion de sigma points
kappa = -1.0

# frecuencia de muestreo
fSampling = 10

### PARAMETROS NO MODIFICABLES

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

for row in vals:
    # obtener delta tiempo
    t1 = row[0] #* 30.0 / 10.0   # si la frecuencia en el preprocesamiento es 10, quitar: *30/10
    dt = t1 - t0

    uk = np.vstack([dt])

    # covarianza proceso
    v = np.array([sigma_vx * np.power(dt, 2) / 2.0, sigma_vx * dt,
                  sigma_vy * np.power(dt, 2) / 2.0, sigma_vy * dt])
    Q = np.outer(v, v)

    # covarianza medicion
    v = np.array([np.power(range_accuracy, 2), np.power(np.deg2rad(angular_resolution), 2)])
    R = np.outer(v, v)
    # R = np.array([[np.power(range_accuracy, 2), 0.0],
    #              [0.0, np.power(np.deg2rad(angular_resolution), 2)]])

    # obtener medicion
    yk = H(np.vstack([row[2], 0.0, row[3], 0.0]))

    # estimar estado con UKF
    xk0, PK0 = UKF_additiveNoise(xk0, PK0, uk, yk, Q, R, F, H,
                                 algo='Julier', kappa=kappa)

    # Mostrar resultados
    '''
    print("Measured output:\n", yk)
    print("Output with estimated state: \n", H(xk0))
    print("Estimated state:\n", xk0)
    print("Q: \n", Q)
    print("R: \n", R)
    print("Covariance P0k:\n", PK0, "\n")
    '''

    # guardar en vectores
    rMea = np.append(rMea, yk[0, 0])
    thetaMea = np.append(thetaMea, yk[1, 0])
    xMea = np.append(xMea, row[2])
    yMea = np.append(yMea, row[3])

    ykEst = H(xk0)
    rEst = np.append(rEst, ykEst[0, 0])
    thetaEst = np.append(thetaEst, ykEst[1, 0])
    xEst = np.append(xEst, xk0[0, 0])
    yEst = np.append(yEst, xk0[2, 0])

    t = np.append(t, t1)

    # actualizar tiempo inicial para dt
    t0 = t1

# plotting

'''
plt.subplot(221)
plt.plot(t, xMea, '-*')
plt.plot(t, xEst, '-*')
plt.legend(('medido', 'estimado'))
plt.title('X')

plt.subplot(222)
plt.plot(t, yMea, '-*')
plt.plot(t, yEst, '-*')
plt.legend(('medido', 'estimado'))
plt.title('Y')

plt.subplot(223)
plt.plot(t, rMea, '-*')
plt.plot(t, rEst, '-*')
plt.legend(('medido', 'estimado'))
plt.title('r')

plt.subplot(224)
plt.plot(t, thetaMea, '-*')
plt.plot(t, thetaEst, '-*')
plt.legend(('medido', 'estimado'))
plt.title('theta')

plt.show()
'''

plt.ion()
plt.subplot(221)
plt.plot(t, xEst, '-*')
# plt.legend(('estimado'))
plt.title('Estimated X')
plt.xlabel('t (s)')
plt.ylabel('X (m)')

plt.subplot(222)
plt.plot(t, yEst, '-*')
# plt.legend(('estimado'))
plt.title('Estimated Y')
plt.xlabel('t (s)')
plt.ylabel('Y (m)')

plt.subplot(223)
plt.plot(t, rMea, '-*')
# plt.legend(('medido'))
plt.title('Measured r')
plt.xlabel('t (s)')
plt.ylabel('r (m)')

plt.subplot(224)
plt.plot(t, thetaMea, '-*')
plt.title(r'Measured $\theta$')
# plt.legend(('medido'))
plt.xlabel('t (s)')
plt.ylabel(r'$\theta$ (rad)')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

plt.ioff()


with rosbag.Bag(bagPath, 'r') as bagFile:
    a = InteractiveBagWithFilter(bagFile, t, xEst, yEst, freq=fSampling)
