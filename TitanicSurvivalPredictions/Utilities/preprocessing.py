import numpy as np
import pandas as pd
from sklearn import preprocessing
from age_prediction import age_classifier
from selectkbest import select_k_best
from quartile_plot import generate_quartile_plot
from feature_importances import feature_importance_scores
from correlation_features import generate_correlation_matrix
from correlation_sort import sorted_correlation
from covariance_sort import sorted_covariance


def random_value(df):
	return int(df.Age.mean() + np.random.randint(-df.Age.std(), high=df.Age.std()))

def preprocessing_df(df):

    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(value=0, inplace=True)
        df[col] = pd.Categorical(df[col], categories=df[col].unique()).codes

    #clf = age_classifier()

    #age_selected_features = ["Pclass", "Parch", "SibSp", "Cabin"]

    # for i in df.index:
    #     if np.isnan(df.at[i, 'Age']):
    #         df.at[i, 'Age'] = clf.predict(df[i:i+1][age_selected_features])[0]
    #df["Age"].fillna(value = random_value(df), inplace=True)
    #df["Age"].fillna(value = 0, inplace=True)

    return df

    
def preprocessing_array(df, selected_features):
    X = np.nan_to_num(preprocessing.scale(np.nan_to_num(df[selected_features])))
    return X
