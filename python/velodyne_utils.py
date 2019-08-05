import numpy as np
import os
import csv
import json
import yaml
import ruamel.yaml
import rosbag
from std_msgs.msg import *
from sensor_msgs.msg import PointCloud, PointCloud2, PointField
from sensor_msgs import point_cloud2 as pcl2
import struct
import subprocess
import math
import copy
import pickle
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
from matplotlib.patches import Ellipse
from shapely.ops import cascaded_union
from shapely.geometry import box, Polygon


def cart2pol(x, y):
    #coordenadas cartesianas a polares
    rho = math.sqrt(x**2 + y**2)
    phi = math.atan2(y, x)
    return [rho, phi]


class lcrCircle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
        
    @classmethod
    def areas(cls, circle1_, circle2_):
        
        
        if circle1_.r > circle2_.r: #r1 > r2
            circle1 = circle1_
            circle2 = circle2_
        else:
            circle2 = circle1_
            circle1 = circle2_
        d2 = (circle1.x-circle2.x)**2 + (circle1.y-circle2.y)**2
        d = math.sqrt(d2)
        r1_2 = circle1.r**2
        r2_2 = circle2.r**2
        Abig = math.pi*r1_2
        Asmall = math.pi*r2_2
        #print r1, r2, d
        if circle1.r + circle2.r < d: # caso en que no se intersectan
            return Asmall, Abig, 0, Abig+Asmall
        if d + 2*circle2.r <= circle1.r: # caso en que una circunferencia esta dentro de la otra
            return Asmall, Abig, Asmall, Abig
        # caso en que se intersectan:
        x = ((r1_2 - r2_2)/d + d) / 2
        #print x
        #print r1_2 - x*x
        y = math.sqrt(r1_2 - x*x)
        #print y
        theta1 = math.asin(y/r1)
        theta2 = math.asin(y/r2)
        Ainter = theta1*r1_2 + theta2*r2_2 - d*y
        Aunion = Abig + Asmall - Ainter
        return Asmall, Abig, Ainter, Aunion



class gaussian_3d:
    
    def __init__(self, mean_ = None, std_ = None, n_ = 0):
        self._mean = np.squeeze(np.asarray(mean_))
        self._std = np.squeeze(np.asarray(std_))
        self._n = n_

    @classmethod
    def create(cls, cluster):
        l = len(cluster)
        if l == 1:
            return cls(np.array(cluster[0][0:3], dtype = 'double'), np.zeros(shape=(1,3), dtype = 'double'), 1)
        X = np.zeros(shape=(3,l), dtype = 'double')
        for i, point in enumerate(cluster):
            X[0, i] = point[0]
            X[1, i] = point[1]
            X[2, i] = point[2]
        return cls(X.mean(axis=1), X.std(axis=1), l)
        
def join_gaussians(g1, g2):
    n_ = g1._n + g2._n
    mean_ = (g1._mean*g1._n + g2._mean*g2._n) / n_
    std_ = np.sqrt( \
        ((g1._std**2+g1._mean**2)*g1._n + (g2._std**2+g2._mean**2)*g2._n) / n_ \
        - mean_**2 \
    )
    return gaussian_3d(mean_, std_, n_)

def dist_gaussians(g1, g2, nd = 3):
    if nd == 3:
        return np.linalg.norm(g1._mean - g2._mean)
    elif nd == 2:
        return np.linalg.norm(g1._mean[0:2] - g2._mean[0:2])
    else:
        return None



#calcula la diferencia normalizada (-180, 180) entre dos angulos [angle1-angle2]
def diff_angles(angle1, angle2):
    #a1 - a2
    d = angle1 - angle2
    if d >= math.pi:
        d -= 2*math.pi
    elif d <= -math.pi:
        d += 2*math.pi
    return d


#calcula el area de interseccion entre dos regiones angulares
def intersect_angles(angles1, angles2):
    #angles: [angle_left, angle_right]
    d_left = diff_angles(angles1[0], angles2[0])
    d_right = diff_angles(angles1[1], angles2[1])
    #print [d_left, d_right]
    if d_left <= 0 and d_right >=0:
        #print "caso 1"
        inter = diff_angles(angles2[1], angles2[0])
    elif d_left >= 0 and d_right <=0:
        #print "caso 2"
        inter = diff_angles(angles1[1], angles1[0])
    elif d_left <= 0 and d_right <=0:
        #print "caso 3"
        inter = diff_angles(angles1[1], angles2[0])
    elif d_left >= 0 and d_right >=0:
        #print "caso 4"
        #print angles2[1]*180/math.pi, "-", angles1[0]*180/math.pi
        inter = diff_angles(angles2[1], angles1[0])
    if inter >= 0:
        return 0
    #esto es debido a que los rangos estan en sentido horario
    return -inter








        

class VelodyneCluster():
    
    def __init__(self, point = None, pars = None):
        #, fn_compare_point_to_cluster_ = VelodyneCluster.fn_compare_point_to_cluster
        self.points = list()
        #self.__compare_point_to_cluster__ = fn_compare_point_to_cluster_
        if point is not None:
            self.points.append(point)
            rho, theta = cart2pol(point[0], point[1])
            self.__first_rho__ = rho
            self.__first_theta__ = theta
            self.__last_rho__ = rho
            self.__last_theta__ = theta
            self.len = 1
        else:  
            self.__first_rho__ = None
            self.__first_theta__ = None
            self.__last_rho__ = None
            self.__last_theta__ = None
            self.len = 0
        self.perimeter = 0
        
        self.gaussian = None
        self.valid = True
        self.line_r2 = None
        
        self.circ_std_factor = 1.5
        self.circle = None
        self.bearing_angle_step = 0.2*math.pi/180 #deberia ser una variable global
        
        self.cov = None
        
        self.delta = 0.1 # este valor debe ser global
        self.poly = None
        self.id = None
        self.pars = pars

        
        
        #g1 = gaussian_3d.create(clusters[7][5])
#g2 = gaussian_3d.create(clusters[7][8])
#g3 = gaussian_3d.create(clusters[7][6])
        
        
        #temporal
        self.__tmp_rho__ = None
        self.__tmp_theta__ = None
        self.__tmp_dist_2d__ = None
        self.__front_check__ = False
        self.__back_check__ = False
        self.__tmp_point__ = None
        
        
    def belong_to_front(self, point):
        self.__front_check__ = False
        #try:
        response = VelodyneCluster.fn_compare_point_to_cluster(self.points[0], 
                                                               self.__first_rho__, self.__first_theta__, 
                                                               point, self.pars)

        if response is None:
            return False
        else:
            self.__front_check__ = True
        self.__tmp_rho__ = response[0]
        self.__tmp_theta__ = response[1]
        self.__tmp_dist_2d__ = response[2]
        self.__tmp_point__ = point

        return self.__front_check__
        #except:
        #    return False
        
    def belong_to_back(self, point):
        self.__back_check__ = False
        #try:
        response = VelodyneCluster.fn_compare_point_to_cluster(self.points[-1], 
                                                               self.__last_rho__, self.__last_theta__, 
                                                               point, self.pars)
        if response is None:
            return False
        else:
            self.__back_check__ = True
        self.__tmp_rho__ = response[0]
        self.__tmp_theta__ = response[1]
        self.__tmp_dist_2d__ = response[2]
        self.__tmp_point__ = point

        return self.__back_check__
        #except:
        #    return False
        
    
    def add_front(self):
        if not self.__front_check__ and len(self.points):
            return
        #debe haber pasado antes por belong_to_front
        if len(self.points) >= 1:
            self.perimeter += self.__tmp_dist_2d__
        
        self.points.insert(0, self.__tmp_point__)
        self.__first_rho__ = self.__tmp_rho__
        self.__first_theta__ = self.__tmp_theta__
        self.len += 1
        self.__front_check__ = False
        
    def add_front_cluster(self, cluster, dist_2d = None):
        self.points = cluster.points + self.points
        self.__first_rho__ = cluster.__first_rho__
        self.__first_theta__ = cluster.__first_theta__

        self.len += cluster.len
        if dist_2d == None:
            dist_2d = math.sqrt((cluster.points[-1][0]-self.points[0][0])**2+
                                (cluster.points[-1][1]-self.points[0][1])**2)
        self.perimeter += cluster.perimeter + dist_2d
            
        
    def add_back_cluster(self, cluster, dist_2d = None):
        self.points += cluster.points
        self.__last_rho__ = cluster.__last_rho__
        self.__last_theta__ = cluster.__last_theta__

        self.len += cluster.len
        if dist_2d == None:
            dist_2d = math.sqrt((cluster.points[0][0]-self.points[-1][0])**2+
                                (cluster.points[0][1]-self.points[-1][1])**2)
        self.perimeter += cluster.perimeter + dist_2d
            
            
    def add_cluster(self, cluster):
        dist1 = math.sqrt((cluster.points[-1][0]-self.points[0][0])**2+
                          (cluster.points[-1][1]-self.points[0][1])**2)
        dist2 = math.sqrt((cluster.points[0][0]-self.points[-1][0])**2+
                          (cluster.points[0][1]-self.points[-1][1])**2)
        if dist1 <= dist2:
            self.add_front_cluster(cluster, dist1)
        else:
            self.add_back_cluster(cluster, dist2)

    def add_back(self):
        if not self.__back_check__ and len(self.points) > 0:
            return
        #debe haber pasado antes por belong_to_back       
        if len(self.points) >= 1:
            self.perimeter += self.__tmp_dist_2d__

        self.points.append(self.__tmp_point__)
        self.__last_rho__ = self.__tmp_rho__
        self.__last_theta__ = self.__tmp_theta__
        self.len += 1
        self.__back_check__ = False
        
    def get_np_points(self, is_z = False):
        x = np.zeros(shape=(self.len))
        y = np.zeros(shape=(self.len))
        if is_z:
            z = np.zeros(shape=(self.len))
        #try:
        for j, point in enumerate(self.points):
            x[j] = point[0]
            y[j] = point[1]
            if is_z:
                z[j] = point[2]
        #except:
        #    print len(self.points)
        if is_z:
            return x, y, z
        return x, y
            
        
    def compute_statistics(self):
        #esto no deberia hacerlo, ya que deberia estar bien, pero en realidad esta funcionando mal
        self.len = len(self.points)
        
        #se actualizan los angulos de barrido pertenecientes al objeto
        self.__first_rho__, self.__first_theta__ = cart2pol(self.points[0][0], self.points[0][1])
        self.__last_rho__, self.__last_theta__ = cart2pol(self.points[-1][0], self.points[-1][1])
        
        
        self.gaussian = gaussian_3d.create(self.points)
        
        if self.len <= 2:
            self.valid = False
            
        if self.perimeter >= 1.2:
            self.valid = False
            
        std_ = np.linalg.norm(self.gaussian._std[0:2])
        if std_ > 2.0:
            self.valid = False
            
        #para verificar si es una linea
        x, y = self.get_np_points()
        if self.len > 7:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            self.line_r2 = r_value**2
            if self.line_r2 > 0.9 and self.line_r2 < 1: # se considera linea
                self.valid = False
        self.cov = np.cov(np.vstack((x,y)))
        
        #calcular el rango minimo y rango maximo
        rmin2 = np.inf
        rmax2 = 0
        for point in self.points:
            r2 = point[0]**2 + point[1]**2
            if r2 < rmin2:
                rmin2 = r2
            elif r2 > rmax2:
                rmax2 = r2
        if self.len == 1:
            rmax2 = rmin2
        self.__rmin__ = math.sqrt(rmin2)
        self.__rmax__ = math.sqrt(rmax2)
        
        return self.valid
    
    def radial_relative_distance(self, cluster): 
        #distancia entre el punto mas cercano de este cluster y el otro
        #asumiendo que se mantiene el "radio" de este cluster
        rho1, theta1 = cart2pol(self.points[-1][0], self.points[-1][1])
        rho2, theta2 = cart2pol(cluster.points[0][0], cluster.points[0][1])
        return rho1*math.sqrt(2*(1-math.cos(theta2-theta1)))
        
    def compute_circ(self):
        #print len(self.points), self.gaussian._std, self.gaussian._mean
        #print self.gaussian, self.gaussian._std[0], self.gaussian._std[1], self.len
        #global std_lcr
        #std_lcr = self.gaussian._std
        if self.gaussian is None or self.len == 1 or (self.gaussian._std[0] == 0 or self.gaussian._std[1] == 0):
            #asumo que tiene solo un elemento
            if self.len == 1:
                x = self.points[0][0]
                y = self.points[0][1]
            else:
                x = self.gaussian._mean[0]
                y = self.gaussian._mean[1]
                
            rho, theta = cart2pol(x, y)
            #paso de un angulo al siguiente en el barrido del laser
            #asumo que el tamano del cluster es igual al paso del angulo de barrido
            #este tamano es proporcional la distancia entre el punto y el laser
            r = rho * math.sqrt(2*(1-math.cos(self.bearing_angle_step)))
        else:
            x = self.gaussian._mean[0]
            y = self.gaussian._mean[1]
            r = math.sqrt(self.gaussian._std[0]**2 + self.gaussian._std[1]**2) * self.circ_std_factor
            
        self.circle = lcrCircle(x, y, r)
        
    def compute_poly(self):
        if self.delta > 0:
            points_xy = []
            for point in self.points:
                p0 = [point[0]-self.delta, point[1]+self.delta]
                p1 = [point[0]+self.delta, point[1]-self.delta]
                p2 = [point[0]+self.delta, point[1]+self.delta]
                p3 = [point[0]-self.delta, point[1]-self.delta]
                #print p0, p1, p2, p3
                points_xy += [p0, p1, p2, p3]
        else:
            points_xy = [[point[0], point[1]] for point in self.points]
        self.poly = Polygon(points_xy).convex_hull
        self.area = self.poly.area
        
        
    @staticmethod
    def intersect_bearing(cluster1, cluster2):
        return intersect_angles([cluster1.__first_theta__, cluster1.__last_theta__],
                               [cluster2.__first_theta__, cluster2.__last_theta__])
    @staticmethod
    def intersect_ranges(cluster1, cluster2):
        ranges1 = [cluster1.__rmin__, cluster1.__rmax__]
        ranges2 = [cluster2.__rmin__, cluster2.__rmax__]
        #print ranges1, ranges2
        d_left = ranges1[0] - ranges2[0]
        d_right = ranges1[1] - ranges2[1]
        if d_left <= 0 and d_right >=0:
            inter = ranges2[1] - ranges2[0]
        elif d_left >= 0 and d_right <=0:
            inter = ranges1[1] - ranges1[0]
        elif d_left <= 0 and d_right <=0:
            inter = ranges1[1] - ranges2[0]
        elif d_left >= 0 and d_right >=0:
            inter = ranges2[1] - ranges1[0]
        if inter <= 0:
            return 0
        return inter
    
    ##@staticmethod, @classmethod
    @staticmethod
    def poly_distance(cluster1, cluster2):
        return cluster1.poly.distance(cluster2.poly)
    @staticmethod
    def poly_intersection(cluster1, cluster2):
        return cluster1.poly.intersection(cluster2.poly).area
    @staticmethod
    def overlap_distance(cluster1, cluster2):
        p_d = VelodyneCluster.poly_distance(cluster1, cluster2)
        if p_d > 1.5:
            return None
        if cluster1.area is None: #tal vez borrar
            cluster1.area = cluster1.poly.area
        if cluster2.area is None: #tal vez borrar
            cluster2.area = cluster2.poly.area
        #min_area = math.min(cluster1.poly.area, cluster2.poly.area)
        area_intersection = VelodyneCluster.poly_intersection(cluster1, cluster2)
        
        area_union = cluster1.area + cluster2.area - area_intersection
        area_union_non_intersected = cascaded_union([cluster1.poly, cluster2.poly]).convex_hull.area
        
        return (area_union_non_intersected/area_union)*(1 - area_intersection/area_union)
        
        #inter = p_i / min_area
        #if inter > 0:
        #    return 1-inter        
        #return 1 + p_d
    @staticmethod
    def max_min_distance(cluster1, cluster2):
        #True if max of min distances < Threshold
        #cluster2 respect to cluster1
        #th_min = 0.2, th_max = 1.5
        for point1 in cluster1.points:
            dmin2 = 1000000
            for point2 in cluster2.points:
                d2 = (point1[0]-point2[0])**2 + (point1[1]-point2[1])**2
                if d2 < 0.35:
                    dmin2 = d2
                    break
            if dmin2 > 1.5:
                return False
        return True
    
    @staticmethod
    def max_min_distance2(cluster1, cluster2):
        #tiene que ser verdadero en al menos un sentido
        return (VelodyneCluster.max_min_distance(cluster1, cluster2) or 
                VelodyneCluster.max_min_distance(cluster1, cluster2))

    @staticmethod
    def fn_compare_point_to_cluster(cluster_point, cluster_rho, cluster_theta, point, pars):
        #global max_two_consecutive_range_diff, factor_dist_angle_relative, nsteps_sep
        __tmp_rho__, __tmp_theta__ = cart2pol(point[0], point[1])

        if abs(cluster_rho-__tmp_rho__) > pars.max_two_consecutive_range_diff:
            #se crea un nuevo cluster si los dos puntos estan muy lejos
            return None
        else:
            rho_m = (cluster_rho+__tmp_rho__)/2
            __tmp_dist_2d__ = (cluster_point[0]-point[0])**2 + (cluster_point[1]-point[1])**2
            dist_3d = math.sqrt(__tmp_dist_2d__ + (cluster_point[2]-point[2])**2)
            __tmp_dist_2d__ = math.sqrt(__tmp_dist_2d__)
            dtheta = abs(__tmp_theta__-cluster_theta)
            if dtheta >= 2*math.pi:
                dtheta -= 2*math.pi
            elif dtheta > math.pi:
                dtheta = 2*math.pi - dtheta
            if dtheta > (0.2*math.pi/180)*pars.nsteps_sep:
                return None
            ddtheta = math.sqrt(2-2*math.cos(dtheta))
            #si la distancia entre dos puntos consecutivos es muy grande entonces se crea un nuevo cluster
            if dist_3d > ddtheta*rho_m*pars.factor_dist_angle_relative:
                return None
        return [__tmp_rho__, __tmp_theta__, __tmp_dist_2d__]

    @staticmethod
    def compare_two_clusters(cluster1, cluster2):
        #cluster2 viene siempre despues de cluster1 en sentido horario
        #se comparan el punto mas a la derecha de cluster1 con el mas a la izquierda de cluster2
        point1 = cluster1.points[-1]
        point2 = cluster2.points[0]
        
        dist_2d = math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
        dist_3d = math.sqrt(dist_2d**2 + (point1[2]-point2[2])**2)
        
        #perimetro de los clusters
        perimeter = cluster1.perimeter + cluster2.perimeter + dist_2d
        
        #return perimeter
    
        #distancia entre los puntos mas cercanos de los dos clusters
        if dist_3d > 0.5:
            return 0
        
        if perimeter > 2:
            return 0
        
        if cluster1.gaussian is None or cluster2.gaussian is None:
            return perimeter
        
        g12 = join_gaussians(cluster1.gaussian, cluster2.gaussian)
        #print g12
        
        #si la distancia entre el nuevo centro y alguno de los puntos es mayor a 1m entonces no se unen los clusters
        for point in cluster1.points+cluster2.points:
            dist_c = math.sqrt((point[0]-g12._mean[0])**2 + (point[1]-g12._mean[1])**2 + (point[2]-g12._mean[2])**2)
            if dist_c > 1:
                return 0
    
        # se retorna el inverso de la desviacion estandar conjunta
        #return 1/np.linalg.norm(g12._std[0:2])
        return perimeter
            

class VelodyneMetaCluster():
    
    
    def __init__(self):
        self.clusters = []
        self.layer = []
        self.index_in_layer = []
        
        self._mean = np.zeros((3), dtype = 'double')
        self._std = np.zeros((3), dtype = 'double')
        
        self.len = 0
        
        self.valid = True
        
    def add_cluster(self, cluster, layer, i = None):
        #print layer, item
        self.clusters.append(cluster)
        self.layer.append(layer)
        if i is not None:
            self.index_in_layer.append(i)
        
    def compute_statistics(self):
        
        points = self.get_points()
            
        n = len(points)
        
        if n == 1:
            self._mean[0] = points[0][0]
            self._mean[1] = points[0][1]
            self._mean[2] = points[0][2]
            self._std[0] = 0
            self._std[1] = 0
            self._std[2] = 0
        else:
            x, y, z = VelodyneMetaCluster.get_np_points(points, True)
            self._mean[0] = x.mean()
            self._mean[1] = y.mean()
            self._mean[2] = z.mean()
            self._std[0] = x.std()
            self._std[1] = y.std()
            self._std[2] = z.std()
            
        self.len = n
        
        #calcular la validez
        
        #que la desviacion estandar sea chica
        if self._std[0] > 0.25 or self._std[1] > 0.25:
            self.valid = False
         
        #que la cantidad de puntos que esten cerca del centro
        #sean mas que cierta cantidad
        npts_near_center = self.len*0.75
        d2max = 0.3**2
        k = 0
        for point in points:
            d2 = (point[0]-self._mean[0])**2 + (point[1]-self._mean[1])**2
            if d2 < d2max:
                k += 1
                if k >= npts_near_center:
                    break
        if k < npts_near_center:
            self.valid = False
            
            
        if n > 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            self.line_r2 = r_value**2
            if self.line_r2 > 0.95 and self.line_r2 < 1: # se considera linea
                self.valid = False
                
                
        #numero de layers presentes y numero de puntos por layer mayor que un umbral
        npoints_per_layer = np.zeros((16), dtype = 'double')
        perimeter_per_layer = np.zeros((16), dtype = 'double')
        #self.clusters
        #self.layer
        #self.index_in_layer
        for i, cluster in enumerate(self.clusters):
            npoints_per_layer[self.layer[i]] += cluster.len
            perimeter_per_layer[self.layer[i]] += cluster.perimeter
        
        cumple = (npoints_per_layer) > 5 & \
                (perimeter_per_layer > 0.3) & (perimeter_per_layer < 1.6)
        
        if np.sum(cumple) < 3:
            self.valid = False
        
    def get_points(self):
        points = []
        
        for cluster in self.clusters:
            #print cluster
            points += cluster.points
        
        return points
        
    def get_xyz_np_points(self, is_z = False):
        
        return VelodyneMetaCluster.get_np_points(self.get_points(), is_z)
        
    @staticmethod
    def get_np_points(points, is_z = False):
        n = len(points)
        x = np.zeros(shape=(n))
        y = np.zeros(shape=(n))
        if is_z:
            z = np.zeros(shape=(n))
        #try:
        for j, point in enumerate(points):
            x[j] = point[0]
            y[j] = point[1]
            if is_z:
                z[j] = point[2]
        if is_z:
            return x, y, z
        return x, y

    def get_detections(self):
        return self._mean.tolist()




def data_pl_k_comp(d1, d2):
    #print d1, d2
    if d1[1] == d2[1]:
        return 0
    if d1[1] > d2[1]:
        return -1
    return 1

def packPoint_velodyne_dets2(points, is_bigendian = False):
    #formato de points es una lista de lista
    #cada punto es una lista de 3 elementos [x, y, z]
    out = ""
    
    endian = "<" #little-endian
    if is_bigendian: #big-endian
        endian = ">"
        
    s = struct.Struct(endian+'f f f')
    for point in points:
        point = tuple(point)
        packed_data = s.pack(*point)
        out += packed_data
    return out

def createPoint_velodyne_det2(points = []):
    pts = PointCloud2()
    pts.fields = [
                  PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                  PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                  PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                 ]
    pts.point_step = 12
    pts.is_dense = True
    pts.is_bigendian = False
    npoints = len(points)
    if npoints>0:
        pts.height = 1
        pts.width = npoints
        pts.row_step = npoints*pts.point_step
        pts.data = packPoint_velodyne_dets2(points, pts.is_bigendian)
    return pts






class VelodyneDetectorParameters:

    def __init__(self):
        self.factor_dist_angle_relative = 12.0
        #parametros / umbrales
        self.max_two_consecutive_range_diff = 0.6#0.35#0.35
        self.factor_dist_angle_relative = 12.0
        self.nsteps_sep = 6
        self.nrows = 16
        

class VelodyneDetector:
    

    
    
    def __init__(self):
        #paso por angulo: 0.2 grados
        self.pars = VelodyneDetectorParameters()
        
        

    def detect(self, msg):

        ncols = msg.width/self.pars.nrows

        indices = {}
        data_pl = []
        for i, k in enumerate(range(-15,16,2)):
            indices[str(k)] = i
            data_pl.append([])
            #clusters.append([])    


        #ETAPA 1: primera ronda de clusters
        #cluster por capa
        for point in pcl2.read_points(msg):
            #se eliminan los puntos sobre dos metros y bajo en piso
            if point[2] > 1.35 or point[2] < -0.65:
                continue
            rho, theta = cart2pol(point[0], point[1])
            phi = math.atan2(point[2], rho)
            theta_deg = round(theta*180/math.pi, 2)
            phi_deg = int(round(phi*180/math.pi, 0))

            #try:
            k = indices[str(phi_deg)]
            #except:
            #    print phi_deg
            #    k = indices[str(phi_deg)]

            data_pl[k].append([point, theta_deg, rho, theta])

        # se ordenan los datos para que por cada capa el angulo barrido
        # vaya de 180 a -180
        for i in range(self.pars.nrows):
            data_pl[k].sort(data_pl_k_comp)

        #aca se empiezan a armar los clusters
        clusters = []
        for data_pl_k in data_pl:
            clusters_i = []
            nex = 0

            for pl in data_pl_k:
                point = pl[0]
                rho = pl[2]
                theta = pl[3]

                belong_to_first_cluster = False
                belong_to_last_cluster = False

                #new = False

                if len(clusters_i) == 0: #caso inicial
                    clusters_i.append(VelodyneCluster(point, self.pars)) #, fn_compare_point_to_cluster, 
                    continue

                belong_to_last_cluster = clusters_i[-1].belong_to_back(point)
                if len(clusters_i)==1 and clusters_i[0].len==1:
                    belong_to_first_cluster = False
                else:
                    belong_to_first_cluster = clusters_i[0].belong_to_front(point)

                if belong_to_last_cluster and belong_to_first_cluster: # se deben unir los dos clusters
                    clusters_i[-1].add_back()
                    if len(clusters_i) > 1:
                        clusters_i[0].add_front_cluster(clusters_i.pop())
                elif belong_to_last_cluster:
                    clusters_i[-1].add_back()
                    continue
                elif not belong_to_first_cluster:
                    #se crea nuevo cluster
                    clusters_i.append(VelodyneCluster(point, self.pars))
                    continue
                else: # belong_to_first_cluster
                    clusters_i[0].add_front()

            for cluster in clusters_i:
                cluster.compute_statistics()

            clusters.append(clusters_i)
        # Fin ETAPA 1

        # ETAPA 2
        # Union de clusters no consecutivos muy cercanos en distancia
        # pertenecientes a la misma capa
        # se calcula la distancia entre los clusters
        for k, clusters_k in enumerate(clusters):
            if len(clusters_k) <= 1:
                continue

            #new_clusters_k = []
            n = len(clusters_k)

            #sim_mtx = np.zeros(shape = (n, n), dtype = 'double')
            #sim_mtx = csr_matrix((n, n), dtype='double')
            sim_mtx = np.full((n, n), 1000, dtype = 'double')
            visited = np.full((n, n), False, dtype = 'bool')
            for i in range(n):
                #llega a 2*2 para dar la vuelta, por eso despues se hace j_%n
                for j_ in range(1+i, 2*n):
                    j = j_%n
                    #sim_mtx[i, j] = 1000
                    #sim_mtx[j, i] = 1000
                    if visited[i, j]:
                        continue
                    rdist = clusters_k[i].radial_relative_distance(clusters_k[j])
                    if rdist > 2:
                        break
                    dist = VelodyneCluster.compare_two_clusters(clusters_k[i], clusters_k[j])
                    #if k==8 and i==2 and j==3:
                    #    print rdist, dist
                    visited[i, j] = True
                    visited[j, i] = True
                    if dist == 0:
                        continue
                    sim_mtx[i, j] = dist
                    sim_mtx[j, i] = dist
                    #print i, j, sim_mtx[i, j]
            #el mismo elemento (i, i) tiene distancia 0
            np.fill_diagonal(sim_mtx, 0)
            #if k == 8:
            #    sim_mtx_ = sim_mtx


            #print k, len(clusters_k)
            db = DBSCAN(metric="precomputed", eps=1.2, min_samples=1, algorithm='brute', 
                        n_jobs=4).fit(sim_mtx)

            labels = db.labels_
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True

            #print labels
            # Number of clusters in labels, ignoring noise if present.
            # en este caso no hay labels = -1, pero de todas formas se incluye
            #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

            unique_labels = set(labels)
            #colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            #print unique_labels
            for label in unique_labels:
                class_member_mask = (labels == label)
                #indices de los clusters que se van a unir
                indices = db.core_sample_indices_[class_member_mask & core_samples_mask]
                if len(indices) <= 1:
                    continue
                #print indices
                for index in indices[1:]:
                    #print indices[0], index, "trying"
                    clusters_k[indices[0]].add_cluster(clusters_k[index])
                    #print indices[0], index, "combined"
                    clusters_k[index] = None
            len_antes = len(clusters_k)
            clusters_k = [cluster for cluster in clusters_k if cluster is not None]
            #print k, len_antes, len(clusters_k) 
            clusters[k] = clusters_k
            for i in range(len(clusters[k])):
                clusters[k][i].compute_statistics()
                clusters[k][i].compute_circ()
                clusters[k][i].compute_poly()
        #fin ETAPA 2

        number_of_clusters_by_layer = [len(c) for c in clusters]
        #print number_of_clusters_by_layer


        number_of_clusters = sum(number_of_clusters_by_layer)
        #print number_of_clusters
        #ETAPA 3
        #se revisan intersecciones en barrido angular y rango
        sim_mtx = np.full((number_of_clusters, number_of_clusters), 1000, dtype = 'double')
        visited = np.full((number_of_clusters, number_of_clusters), False, dtype = 'bool')
        np.fill_diagonal(sim_mtx, 0)
        np.fill_diagonal(visited, True)
        sum_layer = 0
        ok = 0

        #_1 = [ 23, 276, 296]
        #_2 = [ 73, 144, 237, 281]

        #iterando por capa
        for layer, clusters_k in enumerate(clusters[:-1]):
            #iterando por clusters en la k-esima capa
            sum_layer_tmp = sum_layer + number_of_clusters_by_layer[layer]

            for i_, cluster1 in enumerate(clusters_k):
                i = sum_layer + i_

                for j_, cluster2 in enumerate(sum(clusters[layer+1:], [])):
                    j = sum_layer_tmp + j_
                    #cond_ = (i in _1 and j in _2) or (j in _1 and i in _2)

                    if visited[i, j]:
                        continue
                    visited[i, j] = True
                    visited[j, i] = True
                    intersection_bearing = VelodyneCluster.intersect_bearing(cluster1, cluster2)
                    if intersection_bearing == 0:
                        #if cond_:
                        #   print (i, j), "no cumple intersection_bearing"
                        continue
                    intersection_ranges = VelodyneCluster.intersect_ranges(cluster1, cluster2)
                    if cluster1.poly is None:
                        cluster1.compute_poly()
                    if cluster2.poly is None:
                        cluster2.compute_poly()
                    poly_distance = VelodyneCluster.poly_distance(cluster1, cluster2)
                    if intersection_ranges == 0 and poly_distance > 0.35:
                        #if cond_:
                        #    print (i, j), "no cumple intersection_ranges", poly_distance
                        continue
                    #if not VelodyneCluster.max_min_distance2(cluster1, cluster2):
                    #    continue
                    area_distance = VelodyneCluster.overlap_distance(cluster1, cluster2)
                    if area_distance is None:
                        #if cond_:
                        #    print (i, j), "no cumple overlap_distance"
                        continue
                    sim_mtx[i, j] = area_distance
                    sim_mtx[j, i] = area_distance
                    ok += 1

            #for kk, clusters_kk in clusters[k+1:]:

            sum_layer = sum_layer_tmp

        db = DBSCAN(metric="precomputed", eps=0.92, min_samples=1, algorithm='brute', 
                    n_jobs=4).fit(sim_mtx)

        labels = db.labels_
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        i = 0
        all_clusters_layers = [] # lista con los layers de cada cluster
        indices_layer_cluster = []
        for layer, clusters_k in enumerate(clusters):
            indices_layer_cluster.append([])
            for j in range(len(clusters_k)):
                all_clusters_layers.append([layer, j])
                indices_layer_cluster[layer].append(i)
                i += 1


        unique_labels = set(labels)
        metaclusters = []
        # Se crean los meta-clusters
        for label in unique_labels:
            metacluster = VelodyneMetaCluster()
            class_member_mask = (labels == label)
            #indices de los clusters que se van a unir
            indices = db.core_sample_indices_[class_member_mask & core_samples_mask]
            #if label == 11 or label == 48:
            #    print indices
            #print indices
            for index in indices:
                layer = all_clusters_layers[index][0]
                i = all_clusters_layers[index][1]
                #print layer, i
                metacluster.add_cluster(clusters[layer][i], layer, i)
            metacluster.compute_statistics()
            metaclusters.append(metacluster)
            
        detections = [metacluster.get_detections() for metacluster in metaclusters if metacluster.valid]
        
        return detections
