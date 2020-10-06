from sklearn.cluster import DBSCAN
import numpy as np

from utils.constants import IMG_W, IMG_H


def dbscan_on_event_list(ev_list, eps=7, min_samples=10):
    return dbscan_on_list([(e.y, e.x) for e in ev_list], eps=eps, min_samples=min_samples)


def dbscan_on_list(param, eps=7, min_samples=10):
    return DBSCAN(eps=eps, min_samples=min_samples).fit(np.array(param))


def get_cluster_image(events_list, eps=7, min_samples=10, use_unique_events=True):
    if use_unique_events:
        param = [(e.y, e.x) for e in events_list]
        # get unique rows (not values). We need the index of the row, not the values
        _, indexes = np.unique(np.array(param), axis=0, return_index=True)
        events_list = [events_list[i] for i in indexes]
    clusters = dbscan_on_event_list(events_list, eps=eps, min_samples=min_samples)
    return build_img_from_clusters(events_list, clusters)


def build_img_from_clusters(events_list, clusters):
    new_img = np.zeros((IMG_H, IMG_W))
    labels_count = clusters.labels_.max() + 1    # because is zero indexed
    label_prob, label_count = np.zeros(labels_count), np.zeros(labels_count)
    for i, label in enumerate(clusters.labels_):
        if label == -1: continue    # outliers
        event = events_list[i]
        label_prob[label] += event.probs[0]
        label_count[label] += 1
    # get probabilities for each label with a value between 0 and 1
    prob_label = label_prob/(100.0*label_count)
    for i, label in enumerate(clusters.labels_):
        if label == -1: continue    # outliers
        event = events_list[i]
        # label increased by one to difference between label 0 and number 0 in the matrix
        new_img[event.y, event.x] = prob_label[label]
    return new_img
