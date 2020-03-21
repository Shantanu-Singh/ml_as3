import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
import time
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA, TruncatedSVD


def load_wave_data():
    df = pd.read_csv("phphT9Lee.csv")

    X = df.drop(columns=['V22'])

    lb_make = LabelEncoder()
    df['V22'] = lb_make.fit_transform(df['V22'])

    y = df['V22'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

    return X, y, X_train, X_test, y_train, y_test


def base_mlp(X, y):

    start = time.time()

    train_sizes = [50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = MLPClassifier(hidden_layer_sizes=(40,),
                              max_iter=10000,
                              activation='relu',
                              solver='adam',
                              random_state=0)

    title = 'Learning curve for Base MLP Classifier on Wave Data'

    print("Plotting", title)

    train_sizes, train_scores, valid_scores = learning_curve(estimator=estimator, X=X, y=y, train_sizes=train_sizes,
                                                             cv=cv, scoring='neg_mean_squared_error')

    train_scores_mean = -train_scores.mean(axis=1)
    valid_scores_mean = -valid_scores.mean(axis=1)

    end = time.time()

    total = end - start

    print("TOTAL TIME TAKEN : ", total)

    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, marker='.', label='Training error')
    plt.plot(train_sizes, valid_scores_mean, marker='.', label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title(title, fontsize=16, y=1.03)
    plt.legend()
    # plt.ylim(0, )
    plt.savefig('Base_MLP_LC_4.png')
    plt.clf()

    return total


def pca_mlp(X, y):

    start = time.time()

    pca = PCA(n_components=19)
    X_pca = pca.fit_transform(X)

    train_sizes = [50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = MLPClassifier(hidden_layer_sizes=(40,),
                              max_iter=10000,
                              activation='relu',
                              solver='adam',
                              random_state=0)

    title = 'Learning curve for PCA + MLP Classifier on Wave Data'

    print("Plotting", title)

    train_sizes, train_scores, valid_scores = learning_curve(estimator=estimator, X=X_pca, y=y, train_sizes=train_sizes,
                                                             cv=cv, scoring='neg_mean_squared_error')

    train_scores_mean = -train_scores.mean(axis=1)
    valid_scores_mean = -valid_scores.mean(axis=1)

    end = time.time()

    total = end - start

    print("TOTAL TIME TAKEN : ", total)

    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, marker='.', label='Training error')
    plt.plot(train_sizes, valid_scores_mean, marker='.', label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title(title, fontsize=16, y=1.03)
    plt.legend()
    # plt.ylim(0, )
    plt.savefig('PCA_MLP_LC_2.png')
    plt.clf()

    return total

    # mse1 = np.mean((X_train - X_projected)**2)
    # mse = np.sum(mse1)/21
    # print('MSE for dataset 1: ', mse)


def ica_mlp(X, y):

    start = time.time()

    ica = FastICA(n_components=15, max_iter=10000, tol=0.01)
    X_ica = ica.fit_transform(X)
    # print(ica.n_iter_)

    train_sizes = [50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = MLPClassifier(hidden_layer_sizes=(40,),
                              max_iter=10000,
                              activation='relu',
                              solver='adam',
                              random_state=0)

    title = 'Learning curve for ICA + MLP Classifier on Wave Data'

    print("Plotting", title)

    train_sizes, train_scores, valid_scores = learning_curve(estimator=estimator, X=X_ica, y=y, train_sizes=train_sizes,
                                                             cv=cv, scoring='neg_mean_squared_error')

    train_scores_mean = -train_scores.mean(axis=1)
    valid_scores_mean = -valid_scores.mean(axis=1)

    end = time.time()

    total = end - start

    print("TOTAL TIME TAKEN : ", total)

    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, marker='.', label='Training error')
    plt.plot(train_sizes, valid_scores_mean, marker='.', label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title(title, fontsize=16, y=1.03)
    plt.legend()
    # plt.ylim(0, )
    plt.savefig('ICA_MLP_LC_2.png')
    plt.clf()

    return total


def rp_mlp(X, y):

    start = time.time()

    rp = GaussianRandomProjection(n_components=21)
    X_rp = rp.fit_transform(X)

    train_sizes = [50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = MLPClassifier(hidden_layer_sizes=(40,),
                              max_iter=10000,
                              activation='relu',
                              solver='adam',
                              random_state=0)

    title = 'Learning curve for RP + MLP Classifier on Wave Data'

    print("Plotting", title)

    train_sizes, train_scores, valid_scores = learning_curve(estimator=estimator, X=X_rp, y=y, train_sizes=train_sizes,
                                                             cv=cv, scoring='neg_mean_squared_error')

    train_scores_mean = -train_scores.mean(axis=1)
    valid_scores_mean = -valid_scores.mean(axis=1)

    end = time.time()

    total = end - start

    print("TOTAL TIME TAKEN : ", total)

    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, marker='.', label='Training error')
    plt.plot(train_sizes, valid_scores_mean, marker='.', label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title(title, fontsize=16, y=1.03)
    plt.legend()
    # plt.ylim(0, )
    plt.savefig('RP_MLP_LC_2.png')
    plt.clf()

    return total


def tsvd_mlp(X, y):

    start = time.time()

    tsvd = TruncatedSVD(n_components=20)
    X_tsvd = tsvd.fit_transform(X)

    train_sizes = [50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = MLPClassifier(hidden_layer_sizes=(40,),
                              max_iter=10000,
                              activation='relu',
                              solver='adam',
                              random_state=0)

    title = 'Learning curve for TruncatedSVD + MLP Classifier on Wave Data'

    print("Plotting", title)

    train_sizes, train_scores, valid_scores = learning_curve(estimator=estimator, X=X_tsvd, y=y, train_sizes=train_sizes,
                                                             cv=cv, scoring='neg_mean_squared_error')

    train_scores_mean = -train_scores.mean(axis=1)
    valid_scores_mean = -valid_scores.mean(axis=1)

    end = time.time()

    total = end - start

    print("TOTAL TIME TAKEN : ", total)

    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, marker='.', label='Training error')
    plt.plot(train_sizes, valid_scores_mean, marker='.', label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title(title, fontsize=16, y=1.03)
    plt.legend()
    # plt.ylim(0, )
    plt.savefig('TSVD_MLP_LC_2.png')
    plt.clf()

    return total


def km_mlp(X, y):

    start = time.time()

    kmeans = KMeans(n_clusters=2)
    X_km = kmeans.fit_transform(X)

    train_sizes = [50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = MLPClassifier(hidden_layer_sizes=(40,),
                              max_iter=10000,
                              activation='relu',
                              solver='adam',
                              random_state=0)

    title = 'Learning curve for KM + MLP Classifier on Wave Data'

    print("Plotting", title)

    train_sizes, train_scores, valid_scores = learning_curve(estimator=estimator, X=X_km, y=y,
                                                             train_sizes=train_sizes,
                                                             cv=cv, scoring='neg_mean_squared_error')

    train_scores_mean = -train_scores.mean(axis=1)
    valid_scores_mean = -valid_scores.mean(axis=1)

    end = time.time()

    total = end - start

    print("TOTAL TIME TAKEN : ", total)

    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, marker='.', label='Training error')
    plt.plot(train_sizes, valid_scores_mean, marker='.', label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title(title, fontsize=16, y=1.03)
    plt.legend()
    # plt.ylim(0, )
    plt.savefig('KM_MLP_LC_2.png')
    plt.clf()
    return total


def em_mlp(X, y):

    start = time.time()

    em = GaussianMixture(n_components=4, n_init=10, max_iter=500, random_state=0).fit(X)
    X_em = em.predict_proba(X)

    train_sizes = [50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = MLPClassifier(hidden_layer_sizes=(40,),
                              max_iter=10000,
                              activation='relu',
                              solver='adam',
                              random_state=0)

    title = 'Learning curve for EM + MLP Classifier on Wave Data'

    print("Plotting", title)

    train_sizes, train_scores, valid_scores = learning_curve(estimator=estimator, X=X_em, y=y,
                                                             train_sizes=train_sizes,
                                                             cv=cv, scoring='neg_mean_squared_error')

    train_scores_mean = -train_scores.mean(axis=1)
    valid_scores_mean = -valid_scores.mean(axis=1)

    end = time.time()

    total = end - start

    print("TOTAL TIME TAKEN : ", total)

    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, marker='.', label='Training error')
    plt.plot(train_sizes, valid_scores_mean, marker='.', label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title(title, fontsize=16, y=1.03)
    plt.legend()
    # plt.ylim(0, )
    plt.savefig('EM_MLP_LC_2.png')
    plt.clf()

    return total


def plot_accuracy(X_train, X_test, y_train, y_test):

    model = MLPClassifier(hidden_layer_sizes=(40,),
                                  max_iter=10000,
                                  activation='relu',
                                  solver='adam',
                                  random_state=0)
    model.fit(X_train, y_train)
    print("MLP base accuracy : ", model.score(X_test, y_test))

    pca = PCA(n_components=19)
    X_pca_train = pca.fit_transform(X_train)
    X_pca_test = pca.fit_transform(X_test)

    model.fit(X_pca_train, y_train)
    print("PCA + MLP accuracy : ", model.score(X_pca_test, y_test))

    ica = FastICA(n_components=15, max_iter=10000, tol=0.01)
    X_ica_train = ica.fit_transform(X_train)
    X_ica_test = ica.fit_transform(X_test)

    model.fit(X_ica_train, y_train)
    print("ICA + MLP accuracy : ", model.score(X_ica_test, y_test))

    rp = GaussianRandomProjection(n_components=21)
    X_rp_train = rp.fit_transform(X_train)
    X_rp_test = rp.fit_transform(X_test)

    model.fit(X_rp_train, y_train)
    print("RP + MLP accuracy : ", model.score(X_rp_test, y_test))

    tsvd = TruncatedSVD(n_components=20)
    X_tsvd_train = tsvd.fit_transform(X_train)
    X_tsvd_test = tsvd.fit_transform(X_test)

    model.fit(X_tsvd_train, y_train)
    print("TSVD + MLP accuracy : ", model.score(X_tsvd_test, y_test))

    kmeans = KMeans(n_clusters=2)
    X_km_train = kmeans.fit_transform(X_train)
    X_km_test = kmeans.fit_transform(X_test)

    model.fit(X_km_train, y_train)
    print("KM + MLP accuracy : ", model.score(X_km_test, y_test))

    em = GaussianMixture(n_components=4, n_init=10, max_iter=500, random_state=0).fit(X)
    X_em_train = em.predict_proba(X_train)
    X_em_test = em.predict_proba(X_test)

    model.fit(X_em_train, y_train)
    print("EM + MLP accuracy : ", model.score(X_em_test, y_test))


def plot_time(X, y):

    times = []

    times.append(base_mlp(X, y))
    times.append(pca_mlp(X, y))
    times.append(ica_mlp(X, y))
    times.append(rp_mlp(X, y))
    times.append(tsvd_mlp(X, y))
    times.append(km_mlp(X, y))
    times.append(em_mlp(X, y))



if __name__ == "__main__":

    X, y, X_train, X_test, y_train, y_test = load_wave_data()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    plot_time(X_scaled, y)

    plot_accuracy(X_train_scaled, X_test_scaled, y_train, y_test)



