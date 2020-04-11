import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, SelectKBest, chi2

def select_k_best(X, label):
    bestfeatures = SelectKBest(score_func=chi2, k=5)
    fit = bestfeatures.fit(np.nan_to_num(X.drop(label, axis=1)),X[label].astype('int'))
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(np.nan_to_num(X.drop(label, axis=1).columns))
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']
    return featureScores.nlargest(5, 'Score')
