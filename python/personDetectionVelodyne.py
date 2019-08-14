import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rosbag
from std_msgs.msg import *
from sensor_msgs.msg import PointCloud, PointCloud2, PointField
from sensor_msgs import point_cloud2 as pcl2
import math
from scipy import stats
from sklearn.cluster import DBSCAN
from velodyne_utils import VelodyneMetaCluster, VelodyneCluster, VelodyneDetectorParameters, cart2pol, data_pl_k_comp

import pandas as pd
def clustersToCSV(bagfile, name):
    counter = 0
    data = []
    for topic_, msg, t in bagfile.read_messages(topics='/velodyne_points'):
        counter += 1
        print("Count: ", counter, "Time: ", t)
        print(msg.header)

        #############################            SE EJECUTA EL CLUSTERING            #####################################
        ##################################################################################################################
        # distance between consecutive points

        # parametros / umbrales
        nrows = 16
        max_two_consecutive_range_diff = 0.6  # 0.35#0.35
        factor_dist_angle_relative = 12.0
        nsteps_sep = 6
        ncols = msg.width / nrows

        indices = {}
        data_pl = []
        for i, k in enumerate(range(-15, 16, 2)):
            indices[str(k)] = i
            data_pl.append([])

        # ETAPA 1: primera ronda de clusters
        # cluster por capa
        for point in pcl2.read_points(msg):
            # se eliminan los puntos sobre dos metros y bajo en piso
            if point[2] > 1.35 or point[2] < -0.65:
                continue
            rho, theta = cart2pol(point[0], point[1])
            phi = math.atan2(point[2], rho)
            theta_deg = round(theta * 180 / math.pi, 2)
            phi_deg = int(round(phi * 180 / math.pi, 0))

            try:
                k = indices[str(phi_deg)]
            except:
                print(phi_deg)
                k = indices[str(phi_deg)]

            data_pl[k].append([point, theta_deg, rho, theta])

        # se ordenan los datos para que por cada capa el angulo barrido
        # vaya de 180 a -180
        for i in range(nrows):
            data_pl[k].sort(data_pl_k_comp)

        # aca se empiezan a armar los clusters
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

                if len(clusters_i) == 0:  # caso inicial
                    clusters_i.append(VelodyneCluster(point, VelodyneDetectorParameters()))
                    continue

                belong_to_last_cluster = clusters_i[-1].belong_to_back(point)
                if len(clusters_i) == 1 and clusters_i[0].len == 1:
                    belong_to_first_cluster = False
                else:
                    belong_to_first_cluster = clusters_i[0].belong_to_front(point)

                if belong_to_last_cluster and belong_to_first_cluster:  # se deben unir los dos clusters
                    clusters_i[-1].add_back()
                    if len(clusters_i) > 1:
                        clusters_i[0].add_front_cluster(clusters_i.pop())
                elif belong_to_last_cluster:
                    clusters_i[-1].add_back()
                    continue
                elif not belong_to_first_cluster:
                    # se crea nuevo cluster
                    clusters_i.append(VelodyneCluster(point, VelodyneDetectorParameters()))
                    continue
                else:  # belong_to_first_cluster
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

            n = len(clusters_k)

            sim_mtx = np.full((n, n), 1000, dtype='double')
            visited = np.full((n, n), False, dtype='bool')
            for i in range(n):
                # llega a 2*2 para dar la vuelta, por eso despues se hace j_%n
                for j_ in range(1 + i, 2 * n):
                    j = j_ % n
                    if visited[i, j]:
                        continue
                    rdist = clusters_k[i].radial_relative_distance(clusters_k[j])
                    if rdist > 2:
                        break
                    dist = VelodyneCluster.compare_two_clusters(clusters_k[i], clusters_k[j])
                    visited[i, j] = True
                    visited[j, i] = True
                    if dist == 0:
                        continue
                    sim_mtx[i, j] = dist
                    sim_mtx[j, i] = dist
            # el mismo elemento (i, i) tiene distancia 0
            np.fill_diagonal(sim_mtx, 0)

            db = DBSCAN(metric="precomputed", eps=1.2, min_samples=1, algorithm='brute', n_jobs=4).fit(sim_mtx)

            labels = db.labels_
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True

            # Number of clusters in labels, ignoring noise if present.
            # en este caso no hay labels = -1, pero de todas formas se incluye
            unique_labels = set(labels)
            for label in unique_labels:
                class_member_mask = (labels == label)
                # indices de los clusters que se van a unir
                indices = db.core_sample_indices_[class_member_mask & core_samples_mask]
                if len(indices) <= 1:
                    continue
                for index in indices[1:]:
                    clusters_k[indices[0]].add_cluster(clusters_k[index])
                    clusters_k[index] = None
            len_antes = len(clusters_k)
            clusters_k = [cluster for cluster in clusters_k if cluster is not None]
            clusters[k] = clusters_k
            for i in range(len(clusters[k])):
                clusters[k][i].compute_statistics()
                clusters[k][i].compute_circ()
                clusters[k][i].compute_poly()
        # fin ETAPA 2

        number_of_clusters_by_layer = [len(c) for c in clusters]
        print(number_of_clusters_by_layer)
        number_of_clusters = sum(number_of_clusters_by_layer)
        print(number_of_clusters)

        # ETAPA 3
        # se revisan intersecciones en barrido angular y rango
        sim_mtx = np.full((number_of_clusters, number_of_clusters), 1000, dtype='double')
        visited = np.full((number_of_clusters, number_of_clusters), False, dtype='bool')
        np.fill_diagonal(sim_mtx, 0)
        np.fill_diagonal(visited, True)
        sum_layer = 0
        ok = 0

        # iterando por capa
        for layer, clusters_k in enumerate(clusters[:-1]):
            # iterando por clusters en la k-esima capa
            sum_layer_tmp = sum_layer + number_of_clusters_by_layer[layer]

            for i_, cluster1 in enumerate(clusters_k):
                i = sum_layer + i_

                for j_, cluster2 in enumerate(sum(clusters[layer + 1:], [])):
                    j = sum_layer_tmp + j_

                    if visited[i, j]:
                        continue
                    visited[i, j] = True
                    visited[j, i] = True
                    intersection_bearing = VelodyneCluster.intersect_bearing(cluster1, cluster2)
                    if intersection_bearing == 0:
                        continue
                    intersection_ranges = VelodyneCluster.intersect_ranges(cluster1, cluster2)
                    if cluster1.poly is None:
                        cluster1.compute_poly()
                    if cluster2.poly is None:
                        cluster2.compute_poly()
                    poly_distance = VelodyneCluster.poly_distance(cluster1, cluster2)
                    if intersection_ranges == 0 and poly_distance > 0.35:
                        continue
                    area_distance = VelodyneCluster.overlap_distance(cluster1, cluster2)
                    if area_distance is None:
                        continue
                    sim_mtx[i, j] = area_distance
                    sim_mtx[j, i] = area_distance
                    ok += 1

            sum_layer = sum_layer_tmp

        db = DBSCAN(metric="precomputed", eps=0.92, min_samples=1, algorithm='brute', n_jobs=4).fit(sim_mtx)

        labels = db.labels_
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        i = 0
        all_clusters_layers = []  # lista con los layers de cada cluster
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
            # indices de los clusters que se van a unir
            indices = db.core_sample_indices_[class_member_mask & core_samples_mask]
            for index in indices:
                layer = all_clusters_layers[index][0]
                i = all_clusters_layers[index][1]
                metacluster.add_cluster(clusters[layer][i], layer, i)
            metacluster.compute_statistics()
            metaclusters.append(metacluster)

        ##################################################################################################################

        #fig, ax = plt.subplots()
        colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']  #
        axis_ = [-15, 15, -15, 15]
        row = []
        ok = 0
        for i, metacluster in enumerate(metaclusters):
            l = metacluster.len

            if l <= 3:
                continue

            if not metacluster.valid:

                c = 6
            else:
                c = ok % (len(colors) - 1)

                x, y = metacluster.get_xyz_np_points()

                #ax.plot(x, y, colors[c] + '.')
                x_ = x.mean()
                y_ = y.mean()
                if x_ > axis_[0] and x_ < axis_[1] and y_ > axis_[2] and y_ < axis_[3]:
                    row.extend([i, x_, y_])
                    #print("color: ", c, "nro cluster: ", i, "x: ", x_, "y: ", y_)
                    #ax.text(x_ + 0.2, y_ - 0.2, str(i), color=colors[c], fontsize=12)
                ok += 1
        print('ok: ', ok)
        #print("row to append: ", row)
        data.append(row)
        #print(data)
        '''
        ax.set_aspect('equal')
        ax.grid(True, which='both')

        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        plt.axis(axis_)
        plt.pause(0.3)
    plt.show()
    '''
    df = pd.DataFrame(data)
    df.to_csv("clusters_" + name + ".csv")
    return
