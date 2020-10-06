from sklearn.cluster import DBSCAN
import numpy as np

from utils.constants import IMG_W, IMG_H


def dbscan_on_event_list(ev_list, eps=7, min_samples=10):
    return dbscan_on_list([(e.y, e.x) for e in ev_list], eps=eps, min_samples=min_samples)


def dbscan_on_list(param, eps=7, min_samples=10):
    return DBSCAN(eps=eps, min_samples=min_samples).fit(np.array(param))


def get_cluster_image(events_list, eps=7, min_samples=10):
    clusters = dbscan_on_event_list(events_list, eps=eps, min_samples=min_samples)
    return build_img_from_clusters(events_list, clusters)


def build_img_from_clusters(events_list, clusters):
    new_img = np.zeros((IMG_H, IMG_W))
    for i, label in enumerate(clusters.labels_):
        event = events_list[i]
        # label increased by one to difference between label 0 and number 0 in the matrix
        new_img[event.y, event.x] = label + 1
    return new_img
