import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

def feature_importance_scores(X, label):
    model = ExtraTreesClassifier()
    model.fit(np.nan_to_num(X.drop(label, axis=1)),X[label].astype('int'))
    feat_importances = pd.Series(model.feature_importances_, index=X.drop(label, axis=1).columns)
    feat_importances.nlargest(5).plot(kind='barh')
    plt.show()

    return feat_importances