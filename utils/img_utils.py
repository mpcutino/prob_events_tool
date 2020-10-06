import cv2
import numpy as np


def draw_bBox(img, start, end, class_name, prob, bbox_color=(0, 255, 0)):
    """
    draw bounding box in image.
    start is a tuple of (x, y) top left pixel
    end is a tuple of bottom right pixel
    """
    v = cv2.rectangle(img, start, end, bbox_color, 2)

    text = "{0}: {1:.2f}%".format(class_name, prob) if class_name != "" else "{0:.1f}%".format(prob)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)[0]
    cv2.rectangle(img, (start[0], start[1]-text_size[1]), (start[0]+text_size[0], start[1]), bbox_color, cv2.FILLED)
    cv2.putText(img, text, start, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)


def draw_bBox_from_cluster_img(cluster_img, prob_filter=0):
    cv_img = np.array(cluster_img)
    probs = np.unique(cluster_img)
    for p in probs:
        if p == 0: continue
        h_indexes, w_indexes = np.where(cluster_img == p)
        if p >= prob_filter:
            start = w_indexes.min(), h_indexes.min()
            end = w_indexes.max(), h_indexes.max()
            draw_bBox(cv_img, start, end, "", p*100, bbox_color=128/255.0)
        else:
            cv_img[h_indexes, w_indexes] = 0
    return cv_img