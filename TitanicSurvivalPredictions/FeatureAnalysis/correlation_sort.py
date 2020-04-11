import operator
import numpy as np

def sorted_correlation(X, y):
    
    coeff = np.corrcoef(X, y)
    feature_correlation = {}
    i = 0
    for axis in X.axes[0]:
        feature_correlation[axis] = coeff[i, len(X.axes[0])]
        i += 1
    sorted_correlation = sorted(feature_correlation.items(), key=operator.itemgetter(1))

    return sorted_correlation