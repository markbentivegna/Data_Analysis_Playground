import operator
import numpy as np

def sorted_covariance(X, y):
    cov = np.cov(X, y)
    feature_covariance = {}
    i = 0
    for axis in X.axes[0]:
        feature_covariance[axis] = cov[i, len(X.axes[0])]
        i += 1
    sorted_covariance = sorted(feature_covariance.items(), key=operator.itemgetter(1))
    return sorted_covariance