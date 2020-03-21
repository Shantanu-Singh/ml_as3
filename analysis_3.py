import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, v_measure_score, homogeneity_score, completeness_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, mutual_info_score
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
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

    return X, y


def load_vehicle_data():

    df = pd.read_csv("dataset_54_vehicle.csv")

    X = df.drop(columns=['Class'])

    lb_make = LabelEncoder()
    df['Class'] = lb_make.fit_transform(df['Class'])

    y = df['Class'].values

    return X, y


def km_pca(X, y, dataset):

    print("---- KM + PCA ----")

    pca = PCA(0.95)
    X_new = pca.fit_transform(X)

    km_scores = []
    km_silhouette = []
    vmeasure_score = []
    adjusted_rand = []
    mutual_in_score = []
    homogenity = []
    completeness = []

    list_k = list(range(2, 15))
    start = time.time()

    for i in list_k:
        i_start = time.time()

        print("CLUSTER :", i)

        km = KMeans(n_clusters=i, n_init=20, max_iter=500, random_state=0).fit(X_new)
        preds = km.fit_predict(X_new)

        print("KM Score : {}".format(km.score(X_new)))
        km_scores.append(km.score(X_new))

        silhouette = silhouette_score(X_new, preds)
        km_silhouette.append(silhouette)
        print("Silhouette score : {}".format(silhouette))

        ad_rand = adjusted_rand_score(y, preds)
        adjusted_rand.append(ad_rand)
        print("Adjusted random score: {}".format(ad_rand))

        mutual_info = mutual_info_score(y, preds)
        mutual_in_score.append(mutual_info)
        print("Mutual info score : {}".format(mutual_info))

        homo = homogeneity_score(y, preds)
        homogenity.append(homo)
        print("Homogeneity score: {}".format(homo))

        comp = completeness_score(y, preds)
        completeness.append(comp)
        print("Completeness score : {}".format(comp))

        v_measure = v_measure_score(y, preds)
        vmeasure_score.append(v_measure)
        print("V-measure score : {}".format(v_measure))

        i_end = time.time()
        print("Time for this iteration :", (i_end - i_start))

        print("-" * 100)

    end = time.time()

    print("TOTAL TIME", (end - start))

    plt.style.use('seaborn')
    plt.title('KM Clustering on PCA', fontsize=16, y=1.03)
    plt.plot(list_k, km_silhouette, '-o', label='Silhouette score')
    plt.plot(list_k, adjusted_rand, '-o', label='Adjusted Random score')
    plt.plot(list_k, mutual_in_score, '-o', label='Mutual Info score')
    plt.plot(list_k, homogenity, '-o', label='Homogenity score')
    plt.plot(list_k, completeness, '-o', label='Completeness score')
    plt.plot(list_k, vmeasure_score, '-o', label='V-measure score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Metrics score')
    plt.legend()
    filename = 'KM_PCA_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def km_ica(X, y, n, dataset):
    print("---- KM + ICA ----")

    ica = FastICA(n_components=n, max_iter=10000, tol=0.01)
    X_new = ica.fit_transform(X)

    km_scores = []
    km_silhouette = []
    vmeasure_score = []
    adjusted_rand = []
    mutual_in_score = []
    homogenity = []
    completeness = []

    list_k = list(range(2, 15))
    start = time.time()

    for i in list_k:
        i_start = time.time()

        print("CLUSTER :", i)

        km = KMeans(n_clusters=i, n_init=20, max_iter=500, random_state=0).fit(X_new)
        preds = km.fit_predict(X_new)

        print("KM Score : {}".format(km.score(X_new)))
        km_scores.append(km.score(X_new))

        silhouette = silhouette_score(X_new, preds)
        km_silhouette.append(silhouette)
        print("Silhouette score : {}".format(silhouette))

        ad_rand = adjusted_rand_score(y, preds)
        adjusted_rand.append(ad_rand)
        print("Adjusted random score: {}".format(ad_rand))

        mutual_info = mutual_info_score(y, preds)
        mutual_in_score.append(mutual_info)
        print("Mutual info score : {}".format(mutual_info))

        homo = homogeneity_score(y, preds)
        homogenity.append(homo)
        print("Homogeneity score: {}".format(homo))

        comp = completeness_score(y, preds)
        completeness.append(comp)
        print("Completeness score : {}".format(comp))

        v_measure = v_measure_score(y, preds)
        vmeasure_score.append(v_measure)
        print("V-measure score : {}".format(v_measure))

        i_end = time.time()
        print("Time for this iteration :", (i_end - i_start))

        print("-" * 100)

    end = time.time()

    print("TOTAL TIME", (end - start))

    plt.style.use('seaborn')
    plt.title('KM Clustering on ICA', fontsize=16, y=1.03)
    plt.plot(list_k, km_silhouette, '-o', label='Silhouette score')
    plt.plot(list_k, adjusted_rand, '-o', label='Adjusted Random score')
    plt.plot(list_k, mutual_in_score, '-o', label='Mutual Info score')
    plt.plot(list_k, homogenity, '-o', label='Homogenity score')
    plt.plot(list_k, completeness, '-o', label='Completeness score')
    plt.plot(list_k, vmeasure_score, '-o', label='V-measure score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Metrics score')
    plt.legend()
    filename = 'KM_ICA_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def km_rp(X, y, n, dataset):
    print("---- KM + RP ----")

    rp = GaussianRandomProjection(n_components=n)
    X_new = rp.fit_transform(X)

    km_scores = []
    km_silhouette = []
    vmeasure_score = []
    adjusted_rand = []
    mutual_in_score = []
    homogenity = []
    completeness = []

    list_k = list(range(2, 15))
    start = time.time()

    for i in list_k:
        i_start = time.time()

        print("CLUSTER :", i)

        km = KMeans(n_clusters=i, n_init=20, max_iter=500, random_state=0).fit(X_new)
        preds = km.fit_predict(X_new)

        print("KM Score : {}".format(km.score(X_new)))
        km_scores.append(km.score(X_new))

        silhouette = silhouette_score(X_new, preds)
        km_silhouette.append(silhouette)
        print("Silhouette score : {}".format(silhouette))

        ad_rand = adjusted_rand_score(y, preds)
        adjusted_rand.append(ad_rand)
        print("Adjusted random score: {}".format(ad_rand))

        mutual_info = mutual_info_score(y, preds)
        mutual_in_score.append(mutual_info)
        print("Mutual info score : {}".format(mutual_info))

        homo = homogeneity_score(y, preds)
        homogenity.append(homo)
        print("Homogeneity score: {}".format(homo))

        comp = completeness_score(y, preds)
        completeness.append(comp)
        print("Completeness score : {}".format(comp))

        v_measure = v_measure_score(y, preds)
        vmeasure_score.append(v_measure)
        print("V-measure score : {}".format(v_measure))

        i_end = time.time()
        print("Time for this iteration :", (i_end - i_start))

        print("-" * 100)

    end = time.time()

    print("TOTAL TIME", (end - start))

    plt.style.use('seaborn')
    plt.title('KM Clustering on RP', fontsize=16, y=1.03)
    plt.plot(list_k, km_silhouette, '-o', label='Silhouette score')
    plt.plot(list_k, adjusted_rand, '-o', label='Adjusted Random score')
    plt.plot(list_k, mutual_in_score, '-o', label='Mutual Info score')
    plt.plot(list_k, homogenity, '-o', label='Homogenity score')
    plt.plot(list_k, completeness, '-o', label='Completeness score')
    plt.plot(list_k, vmeasure_score, '-o', label='V-measure score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Metrics score')
    plt.legend()
    filename = 'KM_RP_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def km_tsvd(X, y, n, dataset):
    print("---- KM + TSVD ----")

    tsvd = TruncatedSVD(n_components=n)
    X_new = tsvd.fit_transform(X)

    km_scores = []
    km_silhouette = []
    vmeasure_score = []
    adjusted_rand = []
    mutual_in_score = []
    homogenity = []
    completeness = []

    list_k = list(range(2, 15))
    start = time.time()

    for i in list_k:
        i_start = time.time()

        print("CLUSTER :", i)

        km = KMeans(n_clusters=i, n_init=20, max_iter=500, random_state=0).fit(X_new)
        preds = km.fit_predict(X_new)

        print("KM Score : {}".format(km.score(X_new)))
        km_scores.append(km.score(X_new))

        silhouette = silhouette_score(X_new, preds)
        km_silhouette.append(silhouette)
        print("Silhouette score : {}".format(silhouette))

        ad_rand = adjusted_rand_score(y, preds)
        adjusted_rand.append(ad_rand)
        print("Adjusted random score: {}".format(ad_rand))

        mutual_info = mutual_info_score(y, preds)
        mutual_in_score.append(mutual_info)
        print("Mutual info score : {}".format(mutual_info))

        homo = homogeneity_score(y, preds)
        homogenity.append(homo)
        print("Homogeneity score: {}".format(homo))

        comp = completeness_score(y, preds)
        completeness.append(comp)
        print("Completeness score : {}".format(comp))

        v_measure = v_measure_score(y, preds)
        vmeasure_score.append(v_measure)
        print("V-measure score : {}".format(v_measure))

        i_end = time.time()
        print("Time for this iteration :", (i_end - i_start))

        print("-" * 100)

    end = time.time()

    print("TOTAL TIME", (end - start))

    plt.style.use('seaborn')
    plt.title('KM Clustering on TSVD', fontsize=16, y=1.03)
    plt.plot(list_k, km_silhouette, '-o', label='Silhouette score')
    plt.plot(list_k, adjusted_rand, '-o', label='Adjusted Random score')
    plt.plot(list_k, mutual_in_score, '-o', label='Mutual Info score')
    plt.plot(list_k, homogenity, '-o', label='Homogenity score')
    plt.plot(list_k, completeness, '-o', label='Completeness score')
    plt.plot(list_k, vmeasure_score, '-o', label='V-measure score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Metrics score')
    plt.legend()
    filename = 'KM_TSVD_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def km_plot(X, dataset):

    pca = PCA(0.95)
    X_pca = pca.fit_transform(X)

    ica = FastICA(n_components=15, max_iter=10000, tol=0.01)
    X_ica = ica.fit_transform(X)

    rp = GaussianRandomProjection(n_components=21)
    X_rp = rp.fit_transform(X)

    tsvd = TruncatedSVD(n_components=15)
    X_tsvd = tsvd.fit_transform(X)

    km = KMeans(n_clusters=2)
    preds_pca = km.fit_predict(X_pca)
    preds_rp = km.fit_predict(X_rp)
    preds_tsvd = km.fit_predict(X_tsvd)

    km = KMeans(n_clusters=3)
    preds_ica = km.fit_predict(X_ica)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('K-means clustering on dimensionality reduction algorithms', fontsize=12)

    axs[0, 0].set_title('PCA', fontsize=8)
    axs[0, 0].scatter(X[:, 0], X[:, 1], s=50, c=preds_pca, cmap='rainbow')

    axs[0, 1].set_title('ICA', fontsize=8)
    axs[0, 1].scatter(X[:, 0], X[:, 1], s=50, c=preds_ica, cmap='rainbow')

    axs[1, 0].set_title('RP', fontsize=8)
    axs[1, 0].scatter(X[:, 0], X[:, 1], s=50, c=preds_rp, cmap='rainbow')

    axs[1, 1].set_title('TSVD', fontsize=10)
    axs[1, 1].scatter(X[:, 0], X[:, 1], s=50, c=preds_tsvd, cmap='rainbow')

    filename = 'KMeans_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()

    # plt.show()


def em_pca(X, y, dataset):

    print("---- EM + PCA ----")

    pca = PCA(0.95)
    X_new = pca.fit_transform(X)

    em_silhouette = []
    vmeasure_score = []
    adjusted_rand = []
    mutual_in_score = []
    homogenity = []
    completeness = []

    list_k = list(range(2, 15))

    start = time.time()

    for i in list_k:
        i_start = time.time()
        print("CLUSTER :", i)

        em = GaussianMixture(n_components=i, n_init=10, max_iter=500, random_state=0).fit(X_new)
        preds = em.predict(X_new)

        silhouette = silhouette_score(X_new, preds)
        em_silhouette.append(silhouette)
        print("Silhouette score : {}".format(silhouette))

        ad_rand = adjusted_rand_score(y, preds)
        adjusted_rand.append(ad_rand)
        print("Adjusted random score : {}".format(ad_rand))

        mutual_info = mutual_info_score(y, preds)
        mutual_in_score.append(mutual_info)
        print("Adjusted mutual info score : {}".format(mutual_info))

        homo = homogeneity_score(y, preds)
        homogenity.append(homo)
        print("Homogeneity score: {}".format(homo))

        comp = completeness_score(y, preds)
        completeness.append(comp)
        print("Completeness score : {}".format(comp))

        v_measure = v_measure_score(y, preds)
        vmeasure_score.append(v_measure)
        print("V-measure score : {}".format(v_measure))

        print("BIC : {}".format(em.bic(X_new)))
        print("Log-likelihood score : {}".format(em.score(X_new)))

        i_end = time.time()
        print("Time for this iteration :", (i_end - i_start))

        print("-" * 100)

    end = time.time()

    print("TOTAL TIME", (end - start))

    plt.style.use('seaborn')
    plt.title('EM Clustering on PCA', fontsize=16, y=1.03)
    plt.plot(list_k, em_silhouette, '-o', label='Silhouette score')
    plt.plot(list_k, adjusted_rand, '-o', label='Adjusted Random score')
    plt.plot(list_k, mutual_in_score, '-o', label='Mutual Info score')
    plt.plot(list_k, homogenity, '-o', label='Homogenity score')
    plt.plot(list_k, completeness, '-o', label='Completeness score')
    plt.plot(list_k, vmeasure_score, '-o', label='V-measure score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Metrics score')
    plt.legend()
    filename = 'EM_PCA_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def em_ica(X, y, n, dataset):

    print("---- EM + ICA ----")

    ica = FastICA(n_components=n, max_iter=10000, tol=0.01)
    X_new = ica.fit_transform(X)

    em_silhouette = []
    vmeasure_score = []
    adjusted_rand = []
    mutual_in_score = []
    homogenity = []
    completeness = []

    list_k = list(range(2, 15))

    start = time.time()

    for i in list_k:
        i_start = time.time()
        print("CLUSTER :", i)

        em = GaussianMixture(n_components=i, n_init=10, max_iter=500, random_state=0).fit(X_new)
        preds = em.predict(X_new)

        silhouette = silhouette_score(X_new, preds)
        em_silhouette.append(silhouette)
        print("Silhouette score : {}".format(silhouette))

        ad_rand = adjusted_rand_score(y, preds)
        adjusted_rand.append(ad_rand)
        print("Adjusted random score : {}".format(ad_rand))

        mutual_info = mutual_info_score(y, preds)
        mutual_in_score.append(mutual_info)
        print("Adjusted mutual info score : {}".format(mutual_info))

        homo = homogeneity_score(y, preds)
        homogenity.append(homo)
        print("Homogeneity score: {}".format(homo))

        comp = completeness_score(y, preds)
        completeness.append(comp)
        print("Completeness score : {}".format(comp))

        v_measure = v_measure_score(y, preds)
        vmeasure_score.append(v_measure)
        print("V-measure score : {}".format(v_measure))

        print("BIC : {}".format(em.bic(X_new)))
        print("Log-likelihood score : {}".format(em.score(X_new)))

        i_end = time.time()
        print("Time for this iteration :", (i_end - i_start))

        print("-" * 100)

    end = time.time()

    print("TOTAL TIME", (end - start))

    plt.style.use('seaborn')
    plt.title('EM Clustering on ICA', fontsize=16, y=1.03)
    plt.plot(list_k, em_silhouette, '-o', label='Silhouette score')
    plt.plot(list_k, adjusted_rand, '-o', label='Adjusted Random score')
    plt.plot(list_k, mutual_in_score, '-o', label='Mutual Info score')
    plt.plot(list_k, homogenity, '-o', label='Homogenity score')
    plt.plot(list_k, completeness, '-o', label='Completeness score')
    plt.plot(list_k, vmeasure_score, '-o', label='V-measure score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Metrics score')
    plt.legend()
    filename = 'EM_ICA_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def em_rp(X, y, n, dataset):

    print("---- EM + RP ----")

    rp = GaussianRandomProjection(n_components=n)
    X_new = rp.fit_transform(X)

    em_silhouette = []
    vmeasure_score = []
    adjusted_rand = []
    mutual_in_score = []
    homogenity = []
    completeness = []

    list_k = list(range(2, 15))

    start = time.time()

    for i in list_k:
        i_start = time.time()
        print("CLUSTER :", i)

        em = GaussianMixture(n_components=i, n_init=10, max_iter=500, random_state=0).fit(X_new)
        preds = em.predict(X_new)

        silhouette = silhouette_score(X_new, preds)
        em_silhouette.append(silhouette)
        print("Silhouette score : {}".format(silhouette))

        ad_rand = adjusted_rand_score(y, preds)
        adjusted_rand.append(ad_rand)
        print("Adjusted random score : {}".format(ad_rand))

        mutual_info = mutual_info_score(y, preds)
        mutual_in_score.append(mutual_info)
        print("Adjusted mutual info score : {}".format(mutual_info))

        homo = homogeneity_score(y, preds)
        homogenity.append(homo)
        print("Homogeneity score: {}".format(homo))

        comp = completeness_score(y, preds)
        completeness.append(comp)
        print("Completeness score : {}".format(comp))

        v_measure = v_measure_score(y, preds)
        vmeasure_score.append(v_measure)
        print("V-measure score : {}".format(v_measure))

        print("BIC : {}".format(em.bic(X_new)))
        print("Log-likelihood score : {}".format(em.score(X_new)))

        i_end = time.time()
        print("Time for this iteration :", (i_end - i_start))

        print("-" * 100)

    end = time.time()

    print("TOTAL TIME", (end - start))

    plt.style.use('seaborn')
    plt.title('EM Clustering on RP', fontsize=16, y=1.03)
    plt.plot(list_k, em_silhouette, '-o', label='Silhouette score')
    plt.plot(list_k, adjusted_rand, '-o', label='Adjusted Random score')
    plt.plot(list_k, mutual_in_score, '-o', label='Mutual Info score')
    plt.plot(list_k, homogenity, '-o', label='Homogenity score')
    plt.plot(list_k, completeness, '-o', label='Completeness score')
    plt.plot(list_k, vmeasure_score, '-o', label='V-measure score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Metrics score')
    plt.legend()
    filename = 'EM_RP_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def em_tsvd(X, y, n, dataset):

    print("---- EM + TSVD ----")

    tsvd = TruncatedSVD(n_components=n)
    X_new = tsvd.fit_transform(X)

    em_silhouette = []
    vmeasure_score = []
    adjusted_rand = []
    mutual_in_score = []
    homogenity = []
    completeness = []

    list_k = list(range(2, 15))

    start = time.time()

    for i in list_k:
        i_start = time.time()
        print("CLUSTER :", i)

        em = GaussianMixture(n_components=i, n_init=10, max_iter=500, random_state=0).fit(X_new)
        preds = em.predict(X_new)

        silhouette = silhouette_score(X_new, preds)
        em_silhouette.append(silhouette)
        print("Silhouette score : {}".format(silhouette))

        ad_rand = adjusted_rand_score(y, preds)
        adjusted_rand.append(ad_rand)
        print("Adjusted random score : {}".format(ad_rand))

        mutual_info = mutual_info_score(y, preds)
        mutual_in_score.append(mutual_info)
        print("Adjusted mutual info score : {}".format(mutual_info))

        homo = homogeneity_score(y, preds)
        homogenity.append(homo)
        print("Homogeneity score: {}".format(homo))

        comp = completeness_score(y, preds)
        completeness.append(comp)
        print("Completeness score : {}".format(comp))

        v_measure = v_measure_score(y, preds)
        vmeasure_score.append(v_measure)
        print("V-measure score : {}".format(v_measure))

        print("BIC : {}".format(em.bic(X_new)))
        print("Log-likelihood score : {}".format(em.score(X_new)))

        i_end = time.time()
        print("Time for this iteration :", (i_end - i_start))

        print("-" * 100)

    end = time.time()

    print("TOTAL TIME", (end - start))

    plt.style.use('seaborn')
    plt.title('EM Clustering on TSVD', fontsize=16, y=1.03)
    plt.plot(list_k, em_silhouette, '-o', label='Silhouette score')
    plt.plot(list_k, adjusted_rand, '-o', label='Adjusted Random score')
    plt.plot(list_k, mutual_in_score, '-o', label='Mutual Info score')
    plt.plot(list_k, homogenity, '-o', label='Homogenity score')
    plt.plot(list_k, completeness, '-o', label='Completeness score')
    plt.plot(list_k, vmeasure_score, '-o', label='V-measure score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Metrics score')
    plt.legend()
    filename = 'EM_TSVD_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def em_plot(X, dataset):

    pca = PCA(0.95)
    X_pca = pca.fit_transform(X)

    ica = FastICA(n_components=15, max_iter=10000, tol=0.01)
    X_ica = ica.fit_transform(X)

    rp = GaussianRandomProjection(n_components=21)
    X_rp = rp.fit_transform(X)

    tsvd = TruncatedSVD(n_components=15)
    X_tsvd = tsvd.fit_transform(X)

    em = GaussianMixture(n_components=2, n_init=10, max_iter=500, random_state=0)
    preds_pca = em.fit_predict(X_pca)
    preds_rp = em.fit_predict(X_rp)
    preds_tsvd = em.fit_predict(X_tsvd)

    # km = KMeans(n_clusters=3)
    preds_ica = em.fit_predict(X_ica)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('EM clustering on dimensionality reduction algorithms', fontsize=12)

    axs[0, 0].set_title('PCA', fontsize=8)
    axs[0, 0].scatter(X[:, 0], X[:, 1], s=50, c=preds_pca, cmap='rainbow')

    axs[0, 1].set_title('ICA', fontsize=8)
    axs[0, 1].scatter(X[:, 0], X[:, 1], s=50, c=preds_ica, cmap='rainbow')

    axs[1, 0].set_title('RP', fontsize=8)
    axs[1, 0].scatter(X[:, 0], X[:, 1], s=50, c=preds_rp, cmap='rainbow')

    axs[1, 1].set_title('TSVD', fontsize=10)
    axs[1, 1].scatter(X[:, 0], X[:, 1], s=50, c=preds_tsvd, cmap='rainbow')

    filename = 'EM_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()

    # plt.show()


if __name__ == "__main__":
    print('----------Waveform Data----------')

    X, y = load_wave_data()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    km_pca(X_scaled, y, '1')
    km_ica(X_scaled, y, 17, '1')
    km_rp(X_scaled, y, 16, '1')
    km_tsvd(X_scaled, y, 16, '1')

    km_plot(X_scaled, '1')

    em_pca(X, y, '1')
    em_ica(X, y, 17, '1')
    em_rp(X, y, 16, '1')
    em_tsvd(X, y, 16, '1')

    em_plot(X_scaled, '1')

    print('----------Vehicle Data----------')

    X, y = load_vehicle_data()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    km_pca(X_scaled, y, '2')
    km_ica(X_scaled, y, 10, '2')
    km_rp(X_scaled, y, 2, '2')
    km_tsvd(X_scaled, y, 2, '2')

    km_plot(X_scaled, '2')

    em_pca(X, y, '2')
    em_ica(X, y, 10, '2')
    em_rp(X, y, 2, '2')
    em_tsvd(X, y, 2, '2')

    em_plot(X_scaled, '2')


