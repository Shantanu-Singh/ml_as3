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

    return X, y


def load_vehicle_data():

    df = pd.read_csv("dataset_54_vehicle.csv")

    X = df.drop(columns=['Class'])

    lb_make = LabelEncoder()
    df['Class'] = lb_make.fit_transform(df['Class'])

    y = df['Class'].values

    return X, y


def pca_analysis(X, dataset):
    pca = PCA()
    pca.fit_transform(X)

    variance_ratio = pd.Series(pca.explained_variance_ratio_)
    # print(pca.n_components_)

    plt.style.use('seaborn')
    variance_ratio.plot(ylim=(0.0, 0.5), marker='.',  label='Variance ratio')

    plt.xlabel('Number of dimensions')
    plt.ylabel('Explained Variance')
    plt.legend()
    filename = 'PCA_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def ica_analysis(X, dataset):
    ica = FastICA(max_iter=1000, tol=0.0001)
    reduced_X = ica.fit_transform(X)
    order = [-abs(kurtosis(reduced_X[:, i])) for i in range(reduced_X.shape[1])]
    temp = reduced_X[:, np.array(order).argsort()]
    ica_kurt = pd.Series([abs(kurtosis(temp[:, i])) for i in range(temp.shape[1])])

    plt.style.use('seaborn')
    ica_kurt.plot(marker='.')

    plt.xlabel('Independent Components')
    plt.ylabel('Abs Kurtosis - ordered')
    filename = 'ICA_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def tsvd_analysis(X, dataset):

    t_svd = TruncatedSVD(n_components=(X.shape[1] - 1))
    t_svd.fit_transform(X)

    variance_ratio = pd.Series(t_svd.explained_variance_ratio_)

    plt.style.use('seaborn')
    variance_ratio.plot(ylim=(0.0, 0.5), marker='.',  label='Variance ratio')
    # print(variance_ratio)
    # variance.plot(marker='.',  label='Variance')

    plt.xlabel('Number of dimensions')
    plt.ylabel('Explained Variance')
    plt.legend()
    filename = 'TSVD_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def rp_analysis(X, dataset):
    rp = GaussianRandomProjection(n_components=21)
    X_new = rp.fit_transform(X)

    print(X_new.shape)


def calc_mse(X, X_projected):

    mse1 = np.mean((X - X_projected)**2)
    mse = np.sum(mse1)/21
    # print('MSE for dataset 1: ', mse)

    return mse


def reconstruction_error(X):

    pca = PCA(n_components=0.95, svd_solver='full')
    X_pca = pca.fit_transform(X)
    X_pca_proj = pca.inverse_transform(X_pca)

    print(pca.n_components_)

    pca_mse = calc_mse(X, X_pca_proj)

    print("Recontruction error for PCA : ", pca_mse)

    ica = FastICA(n_components=17, tol=0.0001)
    X_ica = ica.fit_transform(X)
    X_ica_proj = ica.inverse_transform(X_ica)
    ica_mse = calc_mse(X, X_ica_proj)

    print("Recontruction error for ICA : ", ica_mse)

    # rp = GaussianRandomProjection(n_components=16)
    # X_rp = rp.fit_transform(X)
    # X_rp_proj = np.matmul(X_rp, rp.components_)
    # rp_mse = calc_mse(X, X_rp_proj)
    #
    # print("Recontruction error for RP : ", rp_mse)

    tsvd = TruncatedSVD(n_components=16)
    X_tsvd = tsvd.fit_transform(X)
    X_tsvd_proj = tsvd.inverse_transform(X_tsvd)
    tsvd_mse = calc_mse(X, X_tsvd_proj)

    print("Recontruction error for TSVD : ", tsvd_mse)





if __name__ == "__main__":
    X, y = load_wave_data()
    pca_analysis(X, '1')
    ica_analysis(X, '1')
    tsvd_analysis(X, '1')
    reconstruction_error(X)

    X, y = load_vehicle_data()
    pca_analysis(X, '2')
    ica_analysis(X, '2')
    tsvd_analysis(X, '2')
    reconstruction_error(X)
