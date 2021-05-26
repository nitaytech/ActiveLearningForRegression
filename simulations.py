import methods
import MACROS as M
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
from time import time
import xgboost

import warnings
import sys
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

# defining the global variables
features = M.AL_FEATURES
y_col = M.AL_Y
train_days = M.AL_TRAIN_DAYS
test_days = M.AL_TEST_DAYS
n_init = M.AL_N_INIT
n_instances = M.AL_N_INSTANCES
n_clusters = M.AL_N_CLUSTERS
bins = M.AL_BINS
classifier_init = LogisticRegression
regressor_init = methods.LinearRegression


from collections.abc import Iterable
import types
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import re



def convert_to_iterable(var):
    vars = var
    if isinstance(vars, str) or not isinstance(vars, Iterable):
        vars = [var]
    return vars



def run_method(regressor, method, list_type, X_train, y_train, X_pool, n_instances):
    """

    :param regressor: regressor init function (For example - LinearRegression (from sklearn.linear_model).
     The fitting (training) is executed inside this function.
    :param method: one of the follows: qbc, emcm, greedy_distances, greedy_predictions, cluster_uncertainty,
    mse_uncertainty, discretization_uncertainty, pareto, greedy_clustering. otherwise - random.
    :param list_type: models or bootstrap. The use is only for qbc and emcm methods.
    :param X_train:
    :param y_train:
    :param X_pool:
    :param n_instances: int. number of instances from the pool to be returned.
    :return: query_idx - indices of the selected instances from the pool. Use X_pool[query_idx].
    """
    learner_list = []
    if list_type == 'models':
        learner_list = methods.list_by_models(X_train, y_train)
    elif list_type == 'bootstrap':
        learner_list = methods.list_by_bootstrap(X_train, y_train, regressor, 5)
    if method == 'qbc':
        query_idx = methods.query_by_committee(learner_list, X_pool, n_instances=n_instances)
    elif method == 'emcm':
        regressor = regressor.fit(X_train, y_train)
        query_idx = methods.expected_model_change_maximization(regressor, learner_list, X_pool, n_instances=n_instances)
    elif method == 'greedy_distances':
        query_idx = methods.greedy_distances(X_train, X_pool, n_instances=n_instances)
    elif method == 'greedy_predictions':
        regressor = regressor.fit(X_train, y_train)
        query_idx = methods.greedy_predictions(regressor, y_train, X_pool, n_instances=n_instances)
    elif method == 'cluster_uncertainty':
        regressor = regressor.fit(X_train, y_train)
        query_idx = methods.cluster_uncertainty(regressor, X_pool, n_clusters=n_clusters, n_instances=n_instances)
    elif method == 'mse_uncertainty':
        regressor = regressor.fit(X_train, y_train)
        query_idx = methods.mse_uncertainty(regressor, X_train, y_train, X_pool, n_instances=n_instances)
    elif method == 'discretization_uncertainty':
        classifier = classifier_init()
        query_idx = methods.discretization_uncertainty(classifier, X_train, y_train, X_pool,
                                                       bins=bins, n_instances=n_instances)
    elif method == 'greedy_paretos':
        g_features = [i for i, col in enumerate(features) if col in M.AL_G_FEATURES]
        b_features = [i for i, col in enumerate(features) if col in M.AL_B_FEATURES]
        query_idx = methods.greedy_paretos(X_train, X_pool, g_features, b_features, n_instances=n_instances)
    elif method == 'pareto':
        g_features = [i for i, col in enumerate(features) if col in M.AL_G_FEATURES]
        b_features = [i for i, col in enumerate(features) if col in M.AL_B_FEATURES]
        query_idx = methods.paretos(X_pool, g_features, b_features, n_instances=n_instances)
    elif method == 'greedy_clustering':
        query_idx = methods.greedy_clustering(X_train, X_pool, n_clusters=n_clusters, n_instances=n_instances)
    else:
        query_idx = random.choices(list(range(X_pool.shape[0])), k=n_instances)
    return query_idx


def simulation(regressor, method, list_type, X_train, y_train, X_pool, y_pool, X_test, y_test):
    query_idx = run_method(regressor, method, list_type, X_train, y_train, X_pool, n_instances)
    X_new = X_pool[query_idx]
    y_new = y_pool[query_idx]
    regressor.fit(np.concatenate([X_train, X_new], axis=0), np.concatenate([y_train, y_new], axis=0))
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return query_idx, mse, mae, r2


def create_init_train(data_train):
    init = data_train.groupby(M.DATE_COLUMN).apply(lambda x: x.sample(n=n_init, random_state=42))
    X_train, y_train = init[features], init[y_col]
    return X_train, y_train


def create_datasets(data):
    train_date = data[M.DATE_COLUMN].min()
    data_train = data[data[M.DATE_COLUMN] <= train_date]
    X_train, y_train = create_init_train(data_train)
    data_val, data_test = train_test_split(data[data[M.DATE_COLUMN] > train_date], test_size=0.05, random_state=42)
    X_test = data_test[features]
    y_test = data_test[y_col]
    return X_train.values, y_train.values, data_val.reset_index(drop=True), X_test.values, y_test.values


def run_methods_simulations(regressor, method, list_type, X_train, y_train, data_val, X_test, y_test, return_train=False):
    results = []
    dates = data_val[M.DATE_COLUMN].unique()
    dates.sort()
    for date in dates:
        filtered_data = data_val[data_val[M.DATE_COLUMN] == date]
        X_pool = filtered_data[features].values
        y_pool = filtered_data[y_col].values
        t0 = time()
        query_idx, mse, mae, r2 = simulation(regressor, method, list_type, X_train, y_train,
                                             X_pool, y_pool, X_test, y_test)
        method_time = (time() - t0) / 60
        X_new = X_pool[query_idx]
        y_new = y_pool[query_idx]
        X_train = np.concatenate([X_train, X_new], axis=0)
        y_train = np.concatenate([y_train, y_new], axis=0)
        query_idx = filtered_data.iloc[query_idx].index.tolist()
        results.append({'regressor': regressor.__class__.__name__ ,'method': method, 'list_type': list_type,
                        'date': date, 'query_idx': query_idx, 'MSE': mse, 'MAE': mae, 'r2': r2, 'time': method_time})
    results = pd.DataFrame(results)
    if return_train:
        return results, X_train, y_train
    else:
        return results


def init_phase(regressor, X_train, y_train, data_val, X_test, y_test, method, steps):
    methods = convert_to_iterable(method)
    train_days = steps
    dates = data_val[M.DATE_COLUMN].unique()
    dates.sort()
    if len(dates) < train_days:
        raise ValueError("Not enough data for init phase - data_val should contain at least " +
                         f"{train_days} dates.")
    results = []
    method_chunk_size = int(np.ceil(train_days / len(methods)))
    for i, method in enumerate(methods):
        method_dates = dates[i * method_chunk_size: min((i+1) * method_chunk_size, train_days)]
        method_data_val = data_val[data_val[M.DATE_COLUMN].isin(method_dates)]
        method_results, X_train, y_train = run_methods_simulations(regressor, method, 'init', X_train, y_train,
                                                                   method_data_val, X_test, y_test, return_train=True)
        results.append(method_results)
    results = pd.concat(results, axis=0)
    used_dates = dates[:train_days]
    return results, X_train, y_train, data_val[data_val[M.DATE_COLUMN].isin(used_dates) == False], X_test, y_test


def run_all_methods(regressor_alg, X_train, y_train, data_val, X_test, y_test):
    results = []
    for method in ['greedy_distances', 'greedy_predictions', 'cluster_uncertainty', 'greedy_paretos',
                   'mse_uncertainty', 'discretization_uncertainty', 'greedy_clustering', 'pareto']:
        regressor = regressor_alg()
        print('\t\t', method)
        results.append(run_methods_simulations(regressor, method, '', X_train,
                                                    y_train, data_val, X_test, y_test))
    for method in ['qbc', 'emcm']:
        for list_type in ['bootstrap', 'models']:
            regressor = regressor_alg()
            print('\t\t', method, list_type)
            results.append(run_methods_simulations(regressor, method, list_type, X_train,
                                                        y_train, data_val, X_test, y_test))
    for i in range(15):
        regressor = regressor_alg()
        results.append(run_methods_simulations(regressor, 'random', str(i), X_train,
                                                    y_train, data_val, X_test, y_test))
    results = pd.concat(results, axis=0)
    return results


def run_methods_with_init(data, regressors, init_method=None, init_steps=None):
    results = []
    init_method_string = '_'.join([str(x) for x in convert_to_iterable(init_method)])
    for regressor_alg in regressors:
        print(regressor_alg.__name__)
        X_train, y_train, data_val, X_test, y_test = create_datasets(data)
        print('\t', init_method_string)
        if init_method is not None and init_steps is not None:
            steps_list = convert_to_iterable(init_steps)
            prev_steps = 0
            prev_steps_results = None
            for steps in steps_list:
                print('\t', steps)
                steps_results = []
                regressor = regressor_alg()
                init_results, X_train, y_train, data_val, X_test, y_test = init_phase(regressor, X_train, y_train,
                                                                                      data_val, X_test, y_test,
                                                                                      init_method, steps - prev_steps)
                if prev_steps_results is not None:
                    init_results = pd.concat([prev_steps_results, init_results], axis=0)
                prev_steps = steps
                prev_steps_results = init_results
                steps_results.append(init_results)
                all_methods_results = run_all_methods(regressor_alg, X_train, y_train, data_val, X_test, y_test)
                steps_results.append(all_methods_results)
                steps_results = pd.concat(steps_results, axis=0)
                steps_results['init'] = init_method_string + '_' + str(steps)
                results.append(steps_results)
        else:
            all_methods_results = run_all_methods(regressor_alg, X_train, y_train, data_val, X_test, y_test)
            all_methods_results['init'] = 'no_init'
            results.append(all_methods_results)
    results = pd.concat(results, axis=0)
    return results


def random_forest():
    return methods.RandomForestRegressor(n_estimators=40, min_samples_leaf=5 ,random_state=42)


def xgboost_regressor():
    return xgboost.XGBRegressor(objective='reg:squarederror', max_depth=20, n_estimators=40, random_state=42)


def mlp_regressor():
    return methods.MLPRegressor(hidden_layer_sizes=(16, 8, ))


def main():
    n_instances = 8
    init_steps = [5, 10, 15, 20, 25]
    init_methods = [None, ['greedy_distances'], ['greedy_paretos'], ['greedy_clustering'], ['random']]
    regressors = [LogisticRegression, random_forest, xgboost_regressor]
    files = []
    for i in range(len(regressors)):
        print(regressors[i].__name__, regressors[i]())
        dfs = []
        for init_method in init_methods:
            data = pd.read_csv(M.DATA_FILE, parse_dates=[M.DATE_COLUMN])
            df = run_methods_with_init(data, regressors=[regressors[i]], init_method=init_method, init_steps=init_steps)
            init_method_string = '_'.join([str(x) for x in convert_to_iterable(init_method)])
            result_file = 'results_k{}_{}_{}.csv'.format(n_instances, regressors[i].__name__, init_method_string)
            df.to_csv(os.path.join(M.RESULTS_FOLDER, result_file), index=False)
            dfs.append(df)
        result_file = 'results_k{}_{}_all.csv'.format(n_instances, regressors[i].__name__)
        pd.concat(dfs, axis=0).to_csv(os.path.join(M.RESULTS_FOLDER, result_file), index=False)
        files.append(os.path.join(M.RESULTS_FOLDER, result_file))
    all_results_df = []
    for file in files:
        all_results_df.append(pd.read_csv(file))
    all_results_df = pd.concat(all_results_df, axis=0)
    all_results_df.to_csv(os.path.join(M.RESULTS_FOLDER, 'results_k{}_all.csv'.format(n_instances)))
