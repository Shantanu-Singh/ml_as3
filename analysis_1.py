import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, v_measure_score, homogeneity_score, completeness_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score, mutual_info_score
from sklearn.mixture import GaussianMixture
import time


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


def kmeans_elbow(X, dataset):
    sse = []
    list_k = list(range(2, 10))

    for k in list_k:
        km = KMeans(n_clusters=k)
        km.fit(X)
        sse.append(km.inertia_)

    # Plot sse against k
    plt.style.use('seaborn')
    plt.figure(figsize=(12, 8))
    plt.plot(list_k, sse, '-o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    filename = 'KM_elbow_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def em_elbow(X_scaled, dataset):
    bic = []
    aic = []
    list_k = list(range(1, 15))

    for k in list_k:
        em = GaussianMixture(n_components=k, n_init=10)
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
    filename = 'EM_elbow_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def kmeans_scores(X_scaled, y, dataset):

    km_scores = []
    km_silhouette = []
    vmeasure_score = []
    adjusted_rand = []
    mutual_in_score = []
    homogenity = []
    completeness = []

    list_k = list(range(2, 30))
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

        i_end = time.time()
        print("Time for this iteration :", (i_end - i_start))

        print("-" * 100)

    end = time.time()

    print("TOTAL TIME", (end - start))

    plt.style.use('seaborn')
    plt.plot(list_k, km_silhouette, '-o', label='Silhouette score')
    plt.plot(list_k, adjusted_rand, '-o', label='Adjusted Random score')
    plt.plot(list_k, mutual_in_score, '-o', label='Mutual Info score')
    plt.plot(list_k, homogenity, '-o', label='Homogenity score')
    plt.plot(list_k, completeness, '-o', label='Completeness score')
    plt.plot(list_k, vmeasure_score, '-o', label='V-measure score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Metrics score')
    plt.legend()
    filename = 'KM_metrics_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def em_scores(X_scaled, y, dataset):

    em_silhouette = []
    vmeasure_score = []
    adjusted_rand = []
    mutual_in_score = []
    homogenity = []
    completeness = []

    list_k = list(range(2, 30))

    start = time.time()

    for i in list_k:
        i_start = time.time()
        print("CLUSTER :", i)
        em = GaussianMixture(n_components=i, n_init=10).fit(X_scaled)
        preds = em.predict(X_scaled)

        silhouette = silhouette_score(X_scaled, preds)
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
    plt.plot(list_k, mutual_in_score, '-o', label='Mutual Info score')
    plt.plot(list_k, homogenity, '-o', label='Homogenity score')
    plt.plot(list_k, completeness, '-o', label='Completeness score')
    plt.plot(list_k, vmeasure_score, '-o', label='V-measure score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Metrics score')
    plt.legend()
    filename = 'EM_metrics_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def km_plot(X, n, dataset):

    km = KMeans(n_clusters=n)
    y_kmeans = km.fit_predict(X)

    plt.title('KM Clustering, 2', fontsize=12, y=1.03)

    plt.scatter(X[:, 0], X[:, 1], s=50, c=y_kmeans, cmap='rainbow')
    filename = 'KM_scatter_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


def em_plot(X, n, dataset):

    em = GaussianMixture(n_components=n, n_init=10)
    y_kmeans = em.fit_predict(X)

    plt.title('EM Clustering, 3', fontsize=12, y=1.03)

    plt.scatter(X[:, 0], X[:, 1], s=50, c=y_kmeans, cmap='rainbow')
    filename = 'EM_scatter_' + dataset + '.png'
    plt.savefig(filename)
    plt.clf()


if __name__ == "__main__":

    print('----------Waveform Data----------')
    X, y = load_wave_data()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans_elbow(X_scaled, '1')
    kmeans_scores(X_scaled, y, '1')
    km_plot(X_scaled, 2, '1')

    em_elbow(X, '1')
    em_scores(X, y, '1')
    em_plot(X_scaled, 3, '1')

    print('----------Vehicle Data----------')

    X, y = load_vehicle_data()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans_elbow(X_scaled, '2')
    kmeans_scores(X_scaled, y, '2')
    km_plot(X_scaled, 2,  '2')

    em_elbow(X, '2')
    em_scores(X, y, '2')
    em_plot(X_scaled, 2, '2')



