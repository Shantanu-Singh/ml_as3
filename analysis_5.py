import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import silhouette_score, v_measure_score, homogeneity_score, completeness_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture
# from time import time
import time
from sklearn.pipeline import Pipeline


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

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=10)
    X_digits_dist = kmeans.fit_transform(X_scaled)
    representative_digit_idx = np.argmin(X_digits_dist, axis=0)
    X_representative_digits = X_scaled[representative_digit_idx]


    train_sizes = [50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = MLPClassifier(hidden_layer_sizes=(100,),
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
    plt.savefig('Base_MLP_LC_3.png')
    plt.clf()



if __name__ == "__main__":
    X, y, X_train, X_test, y_train, y_test = load_wave_data()
    # base_mlp(X, y)

    km_mlp_1(X_train, X_test, y_train, y_test)



