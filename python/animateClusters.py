'''
import rosbag
import sys
from sensor_msgs import point_cloud2 as pcl2
import matplotlib.pyplot as plt

def on_keyboard(event):
    global power
    if event.key == 'right':
        power += 1
    elif event.key == 'left':
        power -= 1

    plt.clf()
    plt.plot(data**power)
    plt.draw()

bagName = '2'

if len(sys.argv) > 1:
    bagName = sys.argv[1]

# open bagFile
bagFile = rosbag.Bag("../data/rawData/" + bagName + ".bag", 'r')

fig = plt.figure()
plt.show()

for topic_, msg, t in bagFile.read_messages(topics='/velodyne_points'):
    X = []
    Y = []
    Z = []
    for point in pcl2.read_points(msg):
        x = point[0]
        y = point[1]
        z = 0
        X.append(x)
        Y.append(y)
        Z.append(z)

    plt.clf()
    plt.plot(X,Y)
    plt.draw()
    raw_input()
'''

from matplotlib import pyplot as plt
import rosbag
import sys
from sensor_msgs import point_cloud2 as pcl2
import numpy as np


class InteractiveBag:
    def __init__(self, bag):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(left=-10, right=10)
        self.ax.set_ylim(bottom=-10, top=10)
        self.line, = self.ax.plot([0], [0], 'o')  # empty line
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.xs = [0]
        self.ys = [0]

        self.sampleFreq = 10.0

        self.bag = bag

        self.counter = 0
        self.updatePoints()
        self.cid = self.line.figure.canvas.mpl_connect('key_press_event', self)
        plt.show()

    def __call__(self, event):
        if event.key == 'right':
            self.counter += 1
            self.updatePoints()
            self.line.figure.canvas.draw()
        elif event.key == 'left' and self.counter - 1 >= 0:
            self.counter -= 1
            self.updatePoints()
            self.line.figure.canvas.draw()

    def updatePoints(self):
        # obtain msg in counter position
        i = 0
        for topic_, msg, t in self.bag.read_messages(topics='/velodyne_points'):
            if i == self.counter:
                break
            i += 1
        # get values
        X = []
        Y = []
        for point in pcl2.read_points(msg):
            x = point[0]
            y = point[1]
            z = point[2]
            if -.5 < z < 1.5 and np.sqrt(np.power(x, 2) + np.power(y, 2)) < 5.5:
                X.append(point[0])
                Y.append(point[1])
        self.xs = X
        self.ys = Y
        self.line.set_data(self.xs, self.ys)
        self.ax.set_title('Frame number ' + str(self.counter))


class InteractiveBagWithFilter:
    def __init__(self, bag, tFilter, xFilter, yFilter, xEst=None, yEst=None, freq=10):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(left=-10, right=10)
        self.ax.set_ylim(bottom=-10, top=10)

        self.mapPoints, = self.ax.plot([0], [0], 'o')
        self.filterPoints, = self.ax.plot([0], [0], 'ro')
        self.drawEstimationPoints = False
        if xEst != None and yEst!=None:
            self.estimationPoints, = self.ax.plot([0], [0], 'go')
            self.drawEstimationPoints = True

        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.xs = [0]
        self.ys = [0]

        self.sampleFreq = freq

        self.bag = bag
        self.bagLen = self.countBagLength()
        self.t = np.linspace(0.0, (self.bagLen - 1) / self.sampleFreq, num=self.bagLen)
        self.filter_x = np.zeros_like(self.t)
        self.filter_y = np.zeros_like(self.t)
        self.fixFilterValues(tFilter, xFilter, yFilter)

        self.counter = 0
        self.updatePoints()
        self.cid = self.mapPoints.figure.canvas.mpl_connect('key_press_event', self)
        plt.show()

    def __call__(self, event):
        if event.key == 'right':
            self.counter += 1
            self.updatePoints()
            self.mapPoints.figure.canvas.draw()
            self.filterPoints.figure.canvas.draw()
        elif event.key == 'left' and self.counter - 1 >= 0:
            self.counter -= 1
            self.updatePoints()
            self.mapPoints.figure.canvas.draw()
            self.filterPoints.figure.canvas.draw()

    def updatePoints(self):
        # obtain msg in counter position
        i = 0
        for topic_, msg, t in self.bag.read_messages(topics='/velodyne_points'):
            if i == self.counter:
                break
            i += 1
        # get values
        X = []
        Y = []
        for point in pcl2.read_points(msg):
            x = point[0]
            y = point[1]
            z = point[2]
            if -.5 < z < 1.5:
                X.append(point[0])
                Y.append(point[1])
        self.xs = X
        self.ys = Y
        self.mapPoints.set_data(self.xs, self.ys)
        self.filterPoints.set_data(self.filter_x[i], self.filter_y[i])
        self.ax.set_title('Frame number ' + str(self.counter) + '. Time: ' + str(self.t[i]))

    def fixFilterValues(self, tFilter, xFilter, yFilter):
        j = 0
        pX = 0.0
        pY = 0.0
        for detectionTime, detectionX, detectionY in zip(tFilter, xFilter, yFilter):
            while j < len(self.t):
                if self.t[j] <= detectionTime:
                    self.filter_x[j] = pX
                    self.filter_y[j] = pY
                else:
                    break
                j += 1
            pX = detectionX
            pY = detectionY
        if len(self.t) - j > 0:
            for k in range(j, len(self.t)):
                self.filter_x[j] = pX
                self.filter_y[j] = pY

    def countBagLength(self):
        i = 0
        for _, _, _ in self.bag.read_messages(topics='/velodyne_points'):
            i += 1
        return i


