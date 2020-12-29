import numba as nb
import numpy as np
from umap.distances import rogers_tanimoto
from numpy.linalg import norm

# -------------------------------------------
# https://github.com/numba/numba/issues/1269
# Workaround current Numba limitation
# in supporting some ufuncs on axis
# -------------------------------------------


@nb.njit
def np_apply_along_axis(func1d, axis, arr):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result


@nb.njit
def np_max(array, axis):
  return np_apply_along_axis(np.max, axis, array)


@nb.njit
def np_min(array, axis):
  return np_apply_along_axis(np.min, axis, array)


# -------------------------------------------------------------------------------------
# Gower Distance
# adapted from: https://github.com/wwwjk366/gower/blob/master/gower/gower_dist.py#L106
# -------------------------------------------------------------------------------------


@nb.njit()
def gower_dist_categorical(a_cat: np.ndarray, b_cat: np.ndarray, weights: np.ndarray) -> np.float:
    """Calculates the Gower Distance for categorical features

    Arguments:
        a_cat {np.ndarray} -- 1d array of features
        b_cat {np.ndarray} -- 2d array of features
        weights {np.ndarray} -- feature weights

    Returns:
        np.float -- gower distance of categorical features
    """
    gower_cat = np.where(a_cat == b_cat,
                         np.zeros_like(a_cat),
                         np.ones_like(a_cat))
    gower_cat = np.multiply(weights, gower_cat).sum()
    return gower_cat


@nb.njit()
def gower_dist_numerical(a_num: np.ndarray, b_num: np.ndarray, weights: np.ndarray) -> np.float:
    """Calculates the Gower Distance for numerical features

    Arguments:
        a_num {np.ndarray} -- 1d array of features
        b_num {np.ndarray} -- 1d array of features
        weights {np.ndarray} -- feature weights

    Returns:
        np.float -- gower distnace of numerical features
    """
    # min-max normalisation
    M = np.vstack((a_num, b_num))
    # assuming no NaN is returned - no missing feature
    max_values = np_max(M, axis=0)  # max per feature
    min_values = np_min(M, axis=0)  # min per feature
    ranges = np.zeros_like(a_num)
    feat_with_range = max_values != 0.0
    ranges[feat_with_range] = 1 - \
        (min_values[feat_with_range] / max_values[feat_with_range])
    # normalise between 0 and 1
    M[:, feat_with_range] /= max_values[feat_with_range]
    # Calculate Gower Distance
    abs_delta = np.absolute(M[0, :] - M[1, :])
    gower_num = np.zeros_like(abs_delta)
    mask_filter = feat_with_range & (ranges != 0)
    gower_num[mask_filter] = abs_delta[mask_filter] / ranges[mask_filter]
    gower_num = np.multiply(weights, gower_num).sum()
    return gower_num



@nb.njit()
def gower_dist_numerical_b(
    a_num: np.ndarray, b_num: np.ndarray,
    weights: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.float:
    """Calculates the Gower Distance for numerical features

    Arguments:
        a_num {np.ndarray} -- 1d array of features
        b_num {np.ndarray} -- 1d array of features
        weights {np.ndarray} -- feature weights
        min_vals {np.ndarray} -- feature sample min value
        max_vals {np.ndarray} -- feature sample max value

    Returns:
        np.float -- gower distance of numerical features
    """
    # assuming no NaN is returned - no missing feature
    ranges = np.abs(max_vals - min_vals)
    feat_w_range = ranges != 0
    # min-max normalisation
    M = np.vstack((a_num, b_num))
    # subtracting min set to 0 var with no range
    # M[:, feat_w_range] -= min_vals[feat_w_range]  #
    M[:, feat_w_range] /= ranges[feat_w_range]
    # Calculate Gower Distance
    M = M[:, feat_w_range]
    gower_num = np.abs(M[1, :] - M[0, :])
    gower_num = np.multiply(weights[feat_w_range], gower_num).sum()
    return gower_num


@nb.njit()
def tanimoto_gower_b(a: np.ndarray, b: np.ndarray,
                    min_vals: np.ndarray,
                    max_vals: np.ndarray,
                    boolean_features: np.ndarray = None,
                    categorical_features: np.ndarray = None,
                    numerical_features: np.ndarray = None,
                    feature_weights: np.ndarray = None) -> np.float:
    """Combined UMAP metric for heterogeneous typed features.
    The metric distance is obtained as the product
    (L2 norm) of the vectors resulting from the
    application of the rogers_tanimoto distance
    (on boolean features), and the gower distance
    (for categorical and numerical features).
    
    modified following the implementation from the R language implementation
    http://search.r-project.org/library/StatMatch/html/gower.dist.html
    categorical-nominal feature should be assigned to cat_gower
    ordinal-feature if number-coded assigned to cat_numerical

    Arguments:
        a {np.ndarray} -- 1d array of features
        b {np.ndarray} -- 1d array of features

    Keyword Arguments:
        boolean_features {np.ndarray} -- array of indices for boolean features (default: {None})
        categorical_features {np.ndarray} -- array of indices for categorical features (default: {None})
        numerical_features {np.ndarray} -- array of indices for numerical features (default: {None})
        feature_weights {np.ndarray} -- array of weights for features (to be used in gower) (default: {None})

    Returns:
        float -- Combined Product metric distance
    """

    bool_dist, gower_dist = 0, 0
    gw_cat_dist, gw_num_dist = 0, 0

    
    boof_in = boolean_features is not None
    catf_in = categorical_features is not None
    numf_in = numerical_features is not None
    
    if feature_weights is None:
        feature_weights = np.ones_like(a)

    if boof_in:
        a_m, b_m = a[boolean_features], b[boolean_features]
        bool_dist = rogers_tanimoto(a_m, b_m)

    if catf_in:
        a_m, b_m = a[categorical_features], b[categorical_features]
        weights_nom = feature_weights[categorical_features]
        gw_cat_dist = gower_dist_categorical(a_m, b_m, weights_nom)

    if numf_in:
        a_m, b_m = a[numerical_features], b[numerical_features]
        weights_num = feature_weights[numerical_features]
        mins = min_vals[numerical_features]
        maxs = max_vals[numerical_features]
        gw_num_dist = gower_dist_numerical_b(a_m, b_m, weights_num, mins, maxs)

    # Complete Gower Distance in case both Numerical and Categorical features are provided
    if catf_in and numf_in:
        weights = np.hstack((weights_nom, weights_num))
        gower_dist = (gw_cat_dist + gw_num_dist) / weights.sum()
    else:
        if catf_in:
            weights = feature_weights[categorical_features]
            gower_dist = gw_cat_dist / weights_nom.sum()
        elif numf_in:
            weights = feature_weights[numerical_features]
            gower_dist = gw_num_dist / weights_num.sum()

    d = np.zeros((2,))
    d[0] = bool_dist
    d[1] = gower_dist
    return norm(d)



@nb.njit()
def tanimoto_gower(a: np.ndarray, b: np.ndarray,
                   boolean_features: np.ndarray = None,
                   categorical_features: np.ndarray = None,
                   numerical_features: np.ndarray = None,
                   feature_weights: np.ndarray = None) -> np.float:
    """Combined UMAP metric for heterogeneous typed features.
    The metric distance is obtained as the product
    (L2 norm) of the vectors resulting from the
    application of the rogers_tanimoto distance
    (on boolean features), and the gower distance
    (for categorical and numerical features).

    Arguments:
        a {np.ndarray} -- 1d array of features
        b {np.ndarray} -- 1d array of features

    Keyword Arguments:
        boolean_features {np.ndarray} -- array of indices for boolean features (default: {None})
        categorical_features {np.ndarray} -- array of indices for categorical features (default: {None})
        numerical_features {np.ndarray} -- array of indices for numerical features (default: {None})
        feature_weights {np.ndarray} -- array of weights for features (to be used in gower) (default: {None})

    Returns:
        float -- Combined Product metric distance
    """

    bool_dist, gower_dist = 0, 0
    gw_cat_dist, gw_num_dist = 0, 0

    boof_in = boolean_features is not None
    catf_in = categorical_features is not None
    numf_in = numerical_features is not None
    
    if feature_weights is None:
        feature_weights = np.ones_like(a)
    

    if boof_in:
        a_m, b_m = a[boolean_features], b[boolean_features]
        bool_dist = rogers_tanimoto(a_m, b_m)

    if catf_in:
        a_m, b_m = a[categorical_features], b[categorical_features]
        weights = feature_weights[categorical_features]
        gw_cat_dist = gower_dist_categorical(a_m, b_m, weights)

    if numf_in:
        a_m, b_m = a[numerical_features], b[numerical_features]
        weights = feature_weights[numerical_features]
        gw_num_dist = gower_dist_numerical(a_m, b_m, weights)

    # Complete Gower Distance in case both Numerical and Categorical features are provided
    if catf_in and numf_in:
        weights = feature_weights[np.hstack(
            (categorical_features, numerical_features))]
        gower_dist = (gw_cat_dist + gw_num_dist) / weights.sum()
    else:
        if catf_in:
            weights = feature_weights[categorical_features]
            gower_dist = gw_cat_dist / weights.sum()
        elif numf_in:
            weights = feature_weights[numerical_features]
            gower_dist = gw_num_dist / weights.sum()

    d = np.zeros((2,))
    d[0] = bool_dist
    d[1] = gower_dist
    return norm(d)
