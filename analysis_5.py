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

    return X, y


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

    # mse1 = np.mean((X_train - X_projected)**2)
    # mse = np.sum(mse1)/21
    # print('MSE for dataset 1: ', mse)


def ica_mlp(X, y):

    start = time.time()

    ica = FastICA(n_components=15)
    X_ica = ica.fit_transform(X)

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


def km_mlp_1(X_train, X_test, y_train, y_test):

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=(40,),
                                  max_iter=10000,
                                  activation='relu',
                                  solver='adam',
                                  random_state=0)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    pipeline = Pipeline([
        ("kmeans", KMeans(n_clusters=3)),
        ("model", MLPClassifier(hidden_layer_sizes=(40,),
                                  max_iter=10000,
                                  activation='relu',
                                  solver='adam',
                                  random_state=0)),
    ])
    pipeline.fit(X_train_scaled, y_train)

    print(pipeline.score(X_test_scaled, y_test))


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



if __name__ == "__main__":

    X, y, = load_wave_data()

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    # base_mlp(X, y)

    # km_mlp_1(X_train, X_test, y_train, y_test)
    pca_mlp(X, y)
    ica_mlp(X, y)
    rp_mlp(X, y)
    tsvd_mlp(X, y)
    km_mlp(X, y)
    em_mlp(X, y)



