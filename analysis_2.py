import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import kurtosis
import numpy as np



def load_wave_data():
    df = pd.read_csv("phphT9Lee.csv")

    X = df.drop(columns=['V22'])

    lb_make = LabelEncoder()
    df['V22'] = lb_make.fit_transform(df['V22'])

    y = df['V22'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

    return X_train, X_test, y_train, y_test, X, y


def pca_analysis(X):
    pca = PCA()
    pca.fit_transform(X)

    variance_ratio = pd.Series(pca.explained_variance_ratio_)
    variance = pd.Series(pca.explained_variance_)
    components = pca.n_components_

    print(variance)
    print(components)

    plt.style.use('seaborn')
    variance_ratio.plot(ylim=(0.0, 0.5), marker='.',  label='Variance ratio')

    plt.xlabel('Number of dimensions')
    plt.ylabel('Explained Variance')
    plt.legend()
    plt.savefig('PCA_1.png')
    plt.clf()


def ica_analysis(X):
    ica = FastICA()
    reduced_X = ica.fit_transform(X)
    order = [-abs(kurtosis(reduced_X[:, i])) for i in range(reduced_X.shape[1])]
    temp = reduced_X[:, np.array(order).argsort()]
    ica_res = pd.Series([abs(kurtosis(temp[:, i])) for i in range(temp.shape[1])])

    # print(reduced_X)
    plt.style.use('seaborn')
    ica_res.plot(marker='.', label='Variance ratio')

    plt.xlabel('Number of dimensions')
    plt.ylabel('Kurtosis')
    plt.legend()
    plt.savefig('ICA_1.png')
    plt.clf()


def tsvd_analysis(X):
    t_svd = TruncatedSVD(n_components=20)
    t_svd.fit_transform(X)

    # ordered_variance

    variance_ratio = pd.Series(t_svd.explained_variance_ratio_)
    # variance = pd.Series(t_svd.explained_variance_)
    # components = t_svd.n_components_

    # print(variance_ratio)
    # print(variance)
    # print(components)

    plt.style.use('seaborn')
    variance_ratio.plot(ylim=(0.0, 0.5), marker='.',  label='Variance ratio')
    # variance.plot(marker='.',  label='Variance')

    plt.xlabel('Number of dimensions')
    plt.ylabel('Explained Variance')
    plt.legend()
    plt.savefig('T-SVD_1.png')
    plt.clf()


def rp_analysis(X):
    rp = GaussianRandomProjection(n_components=21)
    X_new = rp.fit_transform(X)

    print(X_new.shape)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X, y = load_wave_data()
    # pca_analysis(X)
    # ica_analysis(X)
    # tsvd_analysis(X)
    rp_analysis(X)
