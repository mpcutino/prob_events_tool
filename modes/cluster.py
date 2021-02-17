from sklearn.cluster import DBSCAN
import numpy as np


def dbscan_on_event_list(ev_list, eps=7, min_samples=10):
    return dbscan_on_list([(e.y, e.x) for e in ev_list], eps=eps, min_samples=min_samples)


def dbscan_on_list(param, eps=7, min_samples=10):
    return DBSCAN(eps=eps, min_samples=min_samples).fit(np.array(param))


def get_clusters(events_list, pos_of_interest, eps=7, min_samples=10, use_unique_events=True):
    # use only the events from the interest class to build the clusters
    events_of_interest = [e for e in events_list if e.probs[pos_of_interest] == max(e.probs) and e.probs != (0, 100)]
    # events_of_interest = [e for e in events_list if e.probs[pos_of_interest] == max(e.probs)]
    if len(events_of_interest):
        if use_unique_events:
            param = [(e.y, e.x) for e in events_of_interest]
            # get unique rows (not values). We need the index of the row, not the values
            _, indexes = np.unique(np.array(param), axis=0, return_index=True)
            events_of_interest = [events_of_interest[i] for i in indexes]
        clusters = dbscan_on_event_list(events_of_interest, eps=eps, min_samples=min_samples)
        # return build_img_from_clusters(events_of_interest, clusters, events_list)
        return clusters, events_of_interest
    return None, []
