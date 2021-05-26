from modAL.models import ActiveLearner, CommitteeRegressor
from modAL.uncertainty import entropy_sampling
import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import pairwise_distances, silhouette_samples, pairwise_distances_argmin_min
from modAL.disagreement import max_std_sampling
from sklearn.utils import resample
from numpy.linalg import norm


def list_by_models(X, y):
    sklearn_list = [Ridge(), Lasso(), LinearRegression(), RandomForestRegressor(),
                    GradientBoostingRegressor(n_estimators=40, min_samples_leaf=5),
                    KNeighborsRegressor(n_neighbors=2, weights='distance')]
    return [ActiveLearner(estimator=regressor, query_strategy=max_std_sampling, X_training=X, y_training=y)
            for regressor in sklearn_list]


def list_by_bootstrap(X, y, regressor, n):
    models_list = []
    for i in range(n):
        X_s, y_s = resample(X, y)
        models_list.append(ActiveLearner(estimator=clone(regressor), query_strategy=max_std_sampling,
                                         X_training=X_s, y_training=y_s))
    return models_list


def query_by_committee(learner_list, X_pool, n_instances=1):
    committee = CommitteeRegressor(learner_list=learner_list, query_strategy=max_std_sampling)
    query_idx, _ = committee.query(X_pool, n_instances)
    return query_idx


def expected_model_change_maximization(regressor, learner_list, X_pool, n_instances=1):
    changes = []
    for x in X_pool:
        x = x.reshape(1, -1)
        change = 0
        x_norm = norm(x)
        y_hat = regressor.predict(x)[0]
        for learner in learner_list:
            y_p = learner.predict(x)[0]
            change += abs(y_hat - y_p) * x_norm
        change /= len(learner_list)
        changes.append(change)
    query_idx = np.array(changes).argsort()[-n_instances:][::-1]
    return query_idx


def greedy_distances(X_train, X_pool, n_instances=1):
    query_idx = []
    distances = pairwise_distances(X_pool, X_train)
    for i in range(n_instances):
        current_idx = np.argmax(np.min(distances, axis=1))
        query_idx.append(current_idx)
        distances[current_idx, :] = 0
        distances = np.concatenate([distances, pairwise_distances(X_pool, X_pool[current_idx].reshape(1, -1))], axis=1)
    return query_idx


def greedy_predictions(regressor, y_train, X_pool, n_instances=1):
    query_idx = []
    y_pool = regressor.predict(X_pool)
    distances = pairwise_distances(y_pool.reshape(-1, 1), y_train.reshape(-1, 1))
    for i in range(n_instances):
        current_idx = np.argmax(np.min(distances, axis=1))
        query_idx.append(current_idx)
        distances[current_idx, :] = 0
        distances = np.concatenate([distances, pairwise_distances(y_pool.reshape(-1, 1),
                                                                  y_pool[current_idx].reshape(-1, 1))], axis=1)
    return query_idx


def discretization_uncertainty(classifier, X_train, y_train, X_pool, bins, n_instances=1):
    _, bins_edges = np.histogram(y_train, bins)
    d_y_train = np.digitize(y_train, bins_edges)
    learner = ActiveLearner(estimator=classifier, query_strategy=entropy_sampling,
        X_training=X_train, y_training=d_y_train)
    query_idx, _ = learner.query(X_pool, n_instances)
    return query_idx


def mse_uncertainty(regressor, X_train, y_train, X_pool, n_instances=1):
    query_idx = []
    y_pred = regressor.predict(X_train)
    se = (y_train - y_pred) ** 2
    distances = pairwise_distances(X_pool, X_train)
    uncertainty = np.sum(np.multiply(1 / distances, se), axis=1) / np.sum(1 / distances, axis=1)
    for i in range(n_instances):
        current_idx = np.argmax(uncertainty)
        query_idx.append(current_idx)
        uncertainty[current_idx] = 0
    return query_idx


def cluster_uncertainty(regressor, X_pool, n_clusters, n_instances=1):
    query_idx = []
    kmeans = KMeans(n_clusters)
    y_pool = pd.DataFrame(regressor.predict(X_pool), columns=['y'])
    kmeans.fit(X_pool)
    y_pool['cluster'] = kmeans.labels_
    y_pool['silhouette'] = silhouette_samples(y_pool['y'].to_numpy().reshape(-1, 1), y_pool['cluster'])
    selected_clusters = y_pool.groupby('cluster').agg({'y': 'var'}).nlargest(n_instances, 'y').index.tolist()
    for cluster in selected_clusters:
        query_idx.append(y_pool[y_pool['cluster'] == cluster]['silhouette'].idxmin())
    return query_idx


def find_paretos(X_pool, g_features, b_features, minus_gb=False):
    values = X_pool[:, g_features + b_features].copy()
    g_features = list(range(len(g_features)))
    b_features = list(range(len(g_features), values.shape[1]))
    gb_scalar = -1 if minus_gb else 1
    values[:, g_features] = gb_scalar * values[:, g_features]
    values[:, b_features] = -gb_scalar * values[:, b_features]
    is_pareto = np.ones(values.shape[0], dtype=bool)
    for i, values_vec in enumerate(values):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(values[is_pareto] > values_vec, axis=1)  # keep any point with a lower cost
            is_pareto[i] = True  # keep self
    return is_pareto


def paretos(X_pool, g_features=None, b_features=None, n_instances=1):
    query_idx = []
    m, n = X_pool.shape[0], X_pool.shape[1]
    features = list(range(n))
    original_indices = np.arange(m, dtype=np.int)
    if g_features is None and b_features is None:
        g_size = random.randint(2, n - 2) if n > 3 else 1
        g_features = random.sample(features, k=g_size)
        b_features = [f for f in features if f not in g_features]
    elif g_features is None and b_features is not None:
        g_features = [f for f in features if f not in b_features]
    elif g_features is not None and b_features is None:
        b_features = [f for f in features if f not in g_features]
    gb_mask = find_paretos(X_pool, g_features, b_features, minus_gb=False)
    random.seed(42)
    idx = random.randint(0, len(original_indices[gb_mask])-1)
    query_idx.append(original_indices[gb_mask][idx])
    if n_instances == 1:
        return query_idx
    minus_gb_mask = find_paretos(X_pool, g_features, b_features, minus_gb=True)
    random.seed(42)
    idx = random.randint(0, len(original_indices[minus_gb_mask])-1)
    query_idx.append(original_indices[minus_gb_mask][idx])
    if n_instances == 2:
        return query_idx
    X_gb_init = X_pool[query_idx[0], :].reshape(1, -1)
    if n_instances == 3:
        gb_indices = greedy_distances(X_gb_init, X_pool[gb_mask], n_instances=1)
        query_idx += original_indices[gb_mask][gb_indices].tolist()
        return query_idx
    X_minus_gb_init = X_pool[query_idx[1], :].reshape(1, -1)
    gb_indices = greedy_distances(X_gb_init, X_pool[gb_mask], n_instances=int(np.ceil((n_instances - 2)/2)))
    minus_gb_indices = greedy_distances(X_minus_gb_init, X_pool[minus_gb_mask], n_instances=(n_instances - 2)//2)
    query_idx += original_indices[gb_mask][gb_indices].tolist()
    query_idx += original_indices[minus_gb_mask][minus_gb_indices].tolist()
    return query_idx



def greedy_paretos(X_train, X_pool, g_features=None, b_features=None, n_instances=1):
    query_idx = []
    m, n = X_pool.shape[0], X_pool.shape[1]
    features = list(range(n))
    original_indices = np.arange(m, dtype=np.int)
    if g_features is None and b_features is None:
        g_size = random.randint(2, n - 2) if n > 3 else 1
        g_features = random.sample(features, k=g_size)
        b_features = [f for f in features if f not in g_features]
    elif g_features is None and b_features is not None:
        g_features = [f for f in features if f not in b_features]
    elif g_features is not None and b_features is None:
        b_features = [f for f in features if f not in g_features]
    gb_mask = find_paretos(X_pool, g_features, b_features, minus_gb=False)
    minus_gb_mask = find_paretos(X_pool, g_features, b_features, minus_gb=True)
    gb_indices = greedy_distances(X_train, X_pool[gb_mask], n_instances=int(np.ceil((n_instances - 2)/2)))
    query_idx += original_indices[gb_mask][gb_indices].tolist()
    minus_gb_indices = greedy_distances(np.concatenate([X_train, X_pool[query_idx, :]], axis=0),
                                        X_pool[minus_gb_mask], n_instances=(n_instances - 2)//2)
    query_idx += original_indices[minus_gb_mask][minus_gb_indices].tolist()
    return query_idx



def greedy_clustering(X_train, X_pool, n_clusters, n_instances=1):
    kmeans = KMeans(n_clusters, random_state=42)
    kmeans.fit(X_pool)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_pool)
    closest_indices = greedy_distances(X_train, X_pool[closest], n_instances=n_instances)
    query_idx = closest[closest_indices].tolist()
    return query_idx