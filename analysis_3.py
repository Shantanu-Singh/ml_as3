import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, v_measure_score, homogeneity_score, completeness_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
import time
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


def km_pca(X, y):
    pca = PCA(0.95)
    X_new = pca.fit_transform(X)

    km_scores = []
    km_silhouette = []
    vmeasure_score = []
    adjusted_rand = []
    mutual_info_score = []
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

        mutual_info = adjusted_mutual_info_score(y, preds)
        mutual_info_score.append(mutual_info)
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

        i_end = time.time()
        print("Time for this iteration :", (i_end - i_start))

        print("-" * 100)

    end = time.time()

    print("TOTAL TIME", (end - start))

    plt.style.use('seaborn')
    plt.plot(list_k, km_silhouette, '-o', label='Silhouette score')
    plt.plot(list_k, adjusted_rand, '-o', label='Adjusted Random score')
    plt.plot(list_k, mutual_info_score, '-o', label='Adjusted Mutual Info score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Metrics score')
    plt.legend()
    plt.savefig('Optimal_k_3.png')
    plt.clf()

    plt.style.use('seaborn')
    plt.plot(list_k, homogenity, '-o', label='Homogenity score')
    plt.plot(list_k, completeness, '-o', label='Completeness score')
    plt.plot(list_k, vmeasure_score, '-o', label='V-measure score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Metrics score')
    plt.legend()
    plt.savefig('Cluster_quality_3.png')
    plt.clf()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X, y = load_wave_data()
    km_pca(X, y)
