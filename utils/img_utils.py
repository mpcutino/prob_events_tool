import cv2
import numpy as np

from utils.constants import IMG_H, IMG_W


def draw_bBox(img, start, end, class_name, prob, bbox_color=(0, 255, 0)):
    """
    draw bounding box in image.
    start is a tuple of (x, y) top left pixel
    end is a tuple of bottom right pixel
    """
    v = cv2.rectangle(img, start, end, bbox_color, 1)

    text = "{0}: {1:.2f}%".format(class_name, prob) if class_name != "" else "{0:.1f}%".format(prob)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)[0]
    cv2.rectangle(img, (start[0], start[1] - text_size[1]), (start[0] + text_size[0], start[1]), bbox_color, cv2.FILLED)
    cv2.putText(img, text, start, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)


def draw_bBox_from_cluster(cluster, ev_of_interest, all_events, pos_of_interest,
                           prob_filter=0, min_dims=None, use_cluster_prob=False):
    """
    This draw the bounding boxes, but also filter clusters by probability and minimum dimensions
    :param cluster:
    :param ev_of_interest:
    :param all_events:
    :param pos_of_interest:
    :param prob_filter:
    :param use_cluster_prob
    :param min_dims: (height, width) corresponding with the minimum dimensions of the allowed bounding box
    :return:
    """
    cluster_img = build_img_from_clusters(ev_of_interest, cluster, pos_of_interest)
    full_img = build_prob_img(all_events, pos_of_interest)

    probs = np.unique(cluster_img)
    for p in probs:
        if p == 0: continue
        h_indexes, w_indexes = np.where(cluster_img == p)
        start = w_indexes.min(), h_indexes.min()
        end = w_indexes.max(), h_indexes.max()

        box_h = end[1] - start[1]
        box_w = end[0] - start[0]
        h, w = min_dims if min_dims and len(min_dims) == 2 else (box_h, box_w)

        if use_cluster_prob:
            true_prob = p
        else:
            # instead of using prob in cluster (that ony consider elements of the cluster),
            # use the full image in the box slice to compute the probability inside the box
            box_slice = full_img[start[1]:end[1], start[0]:end[0]]
            true_prob = box_slice.sum() / (box_slice > 0).sum()

        if true_prob >= prob_filter and h <= box_h and w <= box_w:
            draw_bBox(full_img, start, end, "", true_prob * 100, bbox_color=128 / 255.0)
        # else:
        #     full_img[h_indexes, w_indexes] = 0
    return full_img


def draw_bBox_from_clusters(cluster, ev_of_interest, all_events, pos_of_interest,
                            prob_filter=0, min_dims=None, use_cluster_prob=False):
    """
    This draw the bounding boxes, but also filter clusters by probability and minimum dimensions
    :param cluster:
    :param ev_of_interest:
    :param all_events:
    :param pos_of_interest:
    :param prob_filter:
    :param use_cluster_prob
    :param min_dims: (height, width) corresponding with the minimum dimensions of the allowed bounding box
    :return:
    """
    # build region proposal from clusters
    rois = get_ROIs(cluster, [(e.x, e.y) for e in ev_of_interest], min_dims)
    # AFTER this, all list must have the same size as ROIS
    # and each position correspond to the rois coords in rois list

    # get the mean and sum probabilities for each class for every region
    rois_probs_sum, rois_probs_count = get_prob_inside_rois(rois, all_events)

    # classify the region using some strategy based on mean and sum values
    strategy = lambda x, y: x       # this strategy selects the sum as the important metric
    strategy = lambda x, y: np.array(x, dtype=np.float)/y    # this strategy selects the mean as the important metric
    classes, probs = classify_roi(rois_probs_sum, rois_probs_count, strategy)

    # filter rois
    filtered_rois = []
    for r, c, p in zip(rois, classes, probs):
        if p > prob_filter and (pos_of_interest is None or pos_of_interest == c):
            filtered_rois.append((r, c, p))

    full_img = build_prob_img(all_events, pos_of_interest)

    for r, c, p in filtered_rois:
        x, y, w, h = r
        draw_bBox(full_img, (x, y), (x+w, y+h), str(c), p * 100, bbox_color=128 / 255.0)
    return full_img


def classify_roi(prob_sum, prob_count, strategy):
    classes = []
    probs = []
    for s, c in zip(prob_sum, prob_count):
        value = strategy(s, c)
        category = np.argmax(value)
        prob = s[category] / c

        classes.append(category)
        probs.append(prob)
    return classes, probs


def get_ROIs(clusters, coord_of_interest, min_dims):
    rois = []
    if clusters is not None:
        all_xy_roi = {}
        for i, label in enumerate(clusters.labels_):
            if label == -1: continue
            x, y = coord_of_interest[i]
            if label not in all_xy_roi:
                all_xy_roi[label] = [], []
            all_xy_roi[label][0].append(x)
            all_xy_roi[label][1].append(y)
        for label in all_xy_roi:
            xs, ys = all_xy_roi[label]
            min_x, min_y = min(xs), min(ys)
            max_x, max_y = max(xs), max(ys)
            w = max_x - min_x
            h = max_y - min_y

            if min_dims and len(min_dims) == 2:
                if w < min_dims[0] or h < min_dims[1]:
                    # too small region
                    continue

            # save minimum x, minimum y, the width and height
            rois.append((min_x, min_y, w, h))
    return rois


def is_in_box(e, (x, y, w, h)):
    return (x <= e.x <= x + w) and (y <= e.y <= y + h)


def get_prob_inside_rois(rois, all_events):
    if len(all_events) and len(rois):
        rois_probs_sum = []
        rois_probs_count = []
        for _ in rois:
            rois_probs_sum.append(np.zeros_like(np.array(all_events[0].probs), dtype=np.float))
            rois_probs_count.append(0)
        for i, box in enumerate(rois):
            for e in all_events:
                if is_in_box(e, box):
                    rois_probs_sum[i] += np.array(e.probs)/100.0
                    rois_probs_count[i] += 1
        return rois_probs_sum, rois_probs_count
    return [], []


def build_img_from_clusters(events_of_interest, clusters, pos_of_interest):
    new_img = np.zeros((IMG_H, IMG_W))
    if clusters is not None:
        labels_count = clusters.labels_.max() + 1  # because is zero indexed
        label_prob, label_count = np.zeros(labels_count), np.zeros(labels_count)
        for i, label in enumerate(clusters.labels_):
            if label == -1: continue  # outliers
            event = events_of_interest[i]
            label_prob[label] += event.probs[pos_of_interest]
            label_count[label] += 1
        # get probabilities for each label with a value between 0 and 1
        prob_label = label_prob / (100.0 * label_count)
        for i, label in enumerate(clusters.labels_):
            if label == -1: continue  # outliers
            event = events_of_interest[i]
            # label increased by one to difference between label 0 and number 0 in the matrix
            new_img[event.y, event.x] = prob_label[label]
    return new_img


def build_prob_img(events, index_of_interest):
    new_image = np.zeros((IMG_H, IMG_W))
    counts = np.zeros((IMG_H, IMG_W))
    for ev in events:
        new_image[ev.y, ev.x] += ev.probs[index_of_interest] / 100.0
        counts[ev.y, ev.x] += 1
    mask = counts > 0
    new_image[mask] = new_image[mask]/counts[mask]
    return new_image


def to3channels(im1, img2):
    background = np.zeros_like(img2)
    background[:im1.shape[0], :im1.shape[1], 0] = im1
    background[:im1.shape[0], :im1.shape[1], 1] = im1
    background[:im1.shape[0], :im1.shape[1], 2] = im1
    return background


def bitwise_img(img1, img2):
    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    # img2_fg = cv2.bitwise_and(img2gray, img2gray, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    return dst
