import matplotlib.pyplot as plt
import seaborn as sns

def generate_correlation_matrix(X):
    corrmat = X.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(11,11))
    g=sns.heatmap(X[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    plt.show()