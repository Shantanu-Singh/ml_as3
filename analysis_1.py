import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, v_measure_score, homogeneity_score, completeness_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture
# from time import time
import time


def load_wave_data():
    df = pd.read_csv("phphT9Lee.csv")

    X = df.drop(columns=['V22'])

    lb_make = LabelEncoder()
    df['V22'] = lb_make.fit_transform(df['V22'])

    y = df['V22'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

    # sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    #
    # for train_index, test_index in sss.split(X, y):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test, X, y


def kmeans_1(X):
    sse = []
    list_k = list(range(2, 15))

    for k in list_k:
        km = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=0)
        km.fit(X)
        sse.append(km.inertia_)

    # Plot sse against k
    plt.style.use('seaborn')
    plt.figure(figsize=(12, 8))
    plt.plot(list_k, sse, '-o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distance')
    plt.savefig('Kmeans_elbow-1.png')
    plt.clf()


def em_1(X_scaled):
    bic = []
    aic = []
    list_k = list(range(1, 11))

    for k in list_k:
        em = GaussianMixture(n_components=k, n_init=10, max_iter=500, random_state=0)
        em.fit(X)
        bic.append(em.bic(X_scaled))
        aic.append(em.aic(X_scaled))

    # Plot sse against k
    plt.style.use('seaborn')
    plt.figure(figsize=(12, 8))
    plt.plot(list_k, bic, '-o', label='BIC')
    plt.plot(list_k, aic, '-o', label='AIC')
    plt.legend()
    plt.xlabel('Number of clusters')
    plt.ylabel('Information Criterion')
    plt.savefig('em_elbow-1.png')
    plt.clf()


def kmeans_scores(X_scaled, y):

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

        km = KMeans(n_clusters=i, n_init=20, max_iter=500, random_state=0).fit(X_scaled)
        preds = km.fit_predict(X_scaled)

        print("KM Score : {}".format(km.score(X_scaled)))
        km_scores.append(km.score(X_scaled))

        silhouette = silhouette_score(X_scaled, preds)
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
    plt.savefig('Optimal_k_1.png')
    plt.clf()

    plt.style.use('seaborn')
    plt.plot(list_k, homogenity, '-o', label='Homogenity score')
    plt.plot(list_k, completeness, '-o', label='Completeness score')
    plt.plot(list_k, vmeasure_score, '-o', label='V-measure score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Metrics score')
    plt.legend()
    plt.savefig('Cluster_quality_1.png')
    plt.clf()



def em_scores(X_scaled, y):

    em_silhouette = []
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
        em = GaussianMixture(n_components=i, n_init=10, max_iter=500, random_state=0).fit(X_scaled)
        preds = em.predict(X_scaled)

        silhouette = silhouette_score(X_scaled, preds)
        em_silhouette.append(silhouette)
        print("Silhouette score : {}".format(silhouette))

        ad_rand = adjusted_rand_score(y, preds)
        adjusted_rand.append(ad_rand)
        print("Adjusted random score : {}".format(ad_rand))

        mutual_info = adjusted_mutual_info_score(y, preds)
        mutual_info_score.append(mutual_info)
        print("Adjusted mutual info score : {}".format(ad_rand))

        homo = homogeneity_score(y, preds)
        homogenity.append(homo)
        print("Homogeneity score: {}".format(homo))

        comp = completeness_score(y, preds)
        completeness.append(comp)
        print("Completeness score : {}".format(comp))

        v_measure = v_measure_score(y, preds)
        vmeasure_score.append(v_measure)
        print("V-measure score : {}".format(v_measure))

        print("BIC : {}".format(em.bic(X_scaled)))
        print("Log-likelihood score : {}".format(em.score(X_scaled)))

        i_end = time.time()
        print("Time for this iteration :", (i_end - i_start))

        print("-" * 100)

    end = time.time()

    print("TOTAL TIME", (end - start))

    plt.style.use('seaborn')
    plt.plot(list_k, em_silhouette, '-o', label='Silhouette score')
    plt.plot(list_k, adjusted_rand, '-o', label='Adjusted Random score')
    plt.plot(list_k, mutual_info_score, '-o', label='Adjusted Mutual Info score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Metrics score')
    plt.legend()
    plt.savefig('EM-Optimal_k_1.png')
    plt.clf()

    plt.style.use('seaborn')
    plt.plot(list_k, homogenity, '-o', label='Homogenity score')
    plt.plot(list_k, completeness, '-o', label='Completeness score')
    plt.plot(list_k, vmeasure_score, '-o', label='V-measure score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Metrics score')
    plt.legend()
    plt.savefig('EM-Cluster_quality_1.png')
    plt.clf()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X, y = load_wave_data()

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    # kmeans_1(X_scaled)
    # em_1(X)

    # kmeans_scores(X_scaled, y)
    em_scores(X, y)
