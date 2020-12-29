from datetime import datetime
from sklearn import metrics
from scipy.spatial.distance import euclidean
# from DBCV import DBCV


def pretty_dict(d, eol_char="\n", title=None):
    """
    Parameters
    ----------
    d: dictionary
    eol_char: string

    Returns
    -------
    dict_print: string
    """
    dict_print = "" if title is None else title + eol_char
    for k, v in d.items():
        dict_print += f"{k:20s} {v:.3f}" + eol_char
    return dict_print


def time_now():
    now = datetime.now()
    now = now.strftime("%b %d %Y %H:%M:%S")
    return str(now)


def update_widget_text(text_widget, new_text, mode="a", widget_eol="<br />"):
    if mode == "a":
        text_widget.text += (widget_eol + new_text)
    elif mode == "w":
        text_widget.text = new_text
    else:
        text_widget.text = new_text
        print(
            "Warning: your selected widget update mode\
             is invalid default with w"
            )


def unsupervised_clu_met(data, labels):
    """
    Parameters
    ----------
    data: numpy.ndarray [n_samples, n_features]
    labels: array-like int or string [n_samples]
    """
    metrics_dict = {}
    metrics_dict['silhouette_score'] = \
        metrics.silhouette_score(data, labels)
    metrics_dict['calinski_harabasz_score'] = \
        metrics.calinski_harabasz_score(data, labels)
    metrics_dict['davies_bouldin_index'] = \
        metrics.davies_bouldin_score(data, labels)
    #metrics_dict["DBCV"] = \
        #DBCV(data, labels, dist_function=euclidean)

    return metrics_dict
