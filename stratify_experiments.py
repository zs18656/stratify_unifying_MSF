import sys
import os
# Add parent directory to Python path
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from strategies import RECMO, DIRMO, DIRREC, Stratify, FixedEnsemble, DynamicStrategy
from TS_functions import getMackey, create_sliding_windows, factors
from copy import deepcopy
import pandas as pd
import numpy as np
from tqdm import tqdm
import os   
import time

def mse(preds, ys):
        return (preds - ys)**2
    
    
def get_horizon_level_stats(strategy_errors):
    stats_df = pd.DataFrame()
    stats_df['mean'] = strategy_errors.mean(axis = 0)
    stats_df['std'] = strategy_errors.std(axis = 0)
    stats_df['min'] = strategy_errors.min(axis = 0)
    stats_df['max'] = strategy_errors.max(axis = 0)
    stats_df['median'] = np.median(strategy_errors, axis = 0)
    stats_df['q1'] = np.quantile(strategy_errors, 0.25, axis = 0)
    stats_df['q3'] = np.quantile(strategy_errors, 0.75, axis = 0)
    return stats_df

def get_stats(preds):
    average_over_forecast = preds.mean(axis = 2).T
    mean_over_strategies = average_over_forecast.mean(axis = 0)
    median_over_strategies = np.median(average_over_forecast, axis = 0)
    upper_quartile = np.percentile(average_over_forecast, 75, axis = 0)
    lower_quartile = np.percentile(average_over_forecast, 25, axis = 0)
    std_over_strategies = np.std(average_over_forecast, axis = 0)
    return mean_over_strategies, median_over_strategies, upper_quartile, lower_quartile, std_over_strategies


def handle_data_splits(univariate_time_series, H_ahead, window_size, train_p, val_p, test_p, verbose = True, tiny = False):

    input_windows, output_windows = create_sliding_windows(univariate_time_series, window_size, H_ahead)
    train_N, val_N, test_N = int(train_p*len(input_windows)), int(val_p*len(input_windows)), int(test_p*len(input_windows))
    forc_xs, forc_ys = input_windows[:train_N], output_windows[:train_N]
    val_xs, val_ys = input_windows[train_N:train_N+val_N], output_windows[train_N:train_N+val_N]
    test_xs, test_ys = input_windows[train_N+val_N:train_N+val_N+test_N], output_windows[train_N+val_N:train_N+val_N+test_N]
    
    if tiny:
        forc_xs, forc_ys = forc_xs[:100], forc_ys[:100]
        val_xs, val_ys = val_xs[:100], val_ys[:100]
        test_xs, test_ys = test_xs[:100], test_ys[:100]

    if verbose:
        print(f"Train data shape: {forc_xs.shape, forc_ys.shape}")
        print(f"Validation data shape: {val_xs.shape, val_ys.shape}")
        print(f"Test data shape: {test_xs.shape, test_ys.shape}")
    return forc_xs, forc_ys, val_xs, val_ys, test_xs, test_ys

def fit_fixed_strategies(forc_xs, forc_ys, forecasting_function, H_ahead, strategy_types, 
                   directory = None, verbose = True):
    strategy_list = []
    strategy_names = []
    fit_time_dict = {}

    for s in factors(H_ahead)[:-1]:
        if 'RECMO' in strategy_types:
            strategy_list.append(RECMO(deepcopy(forecasting_function), H_ahead, s))
            strategy_names.append(f"RECMO_{s}")
        if 'DIRMO' in strategy_types:
            strategy_list.append(DIRMO(deepcopy(forecasting_function), H_ahead, s))
            strategy_names.append(f"DIRMO_{s}")
        if 'DIRREC' in strategy_types:
            strategy_list.append(DIRREC(deepcopy(forecasting_function), H_ahead, s))
            strategy_names.append(f"DIRREC_{s}")
    
    s = H_ahead
    strategy_list.append(RECMO(deepcopy(forecasting_function), H_ahead, s))
    strategy_names.append(f"RECMO_{s}")  
    
    n_base_strategies = len(strategy_list)
    
    print('fitting base strategies')
    # assert False, f'{directory}'
    for idx, x in tqdm(enumerate(strategy_list[:n_base_strategies])):
        start_time = time.time()
        x.fit(forc_xs, forc_ys, save_location = directory)
        end_time = time.time()
        fit_time_dict[strategy_names[idx]] = end_time - start_time

    n_single_strategies = len(strategy_list)
    
    if 'STRATIFY' in strategy_types:
        for i in range(n_single_strategies):
            for j in range(n_single_strategies):
                pair = (strategy_names[i], strategy_names[j])
                # for stratify, we can only copy the base model. deep copy for residual forecaster because we dont want to reassign weights of base models
                strategy_list.append(Stratify(strategy_list[i], deepcopy(strategy_list[j]), deepcopy(strategy_list[j]), H_ahead))  
                strategy_names.append(f"stratify{pair[0]}_{pair[1]}")
                
    assert len(strategy_names) == len(strategy_list)

    if verbose:
        print(strategy_names)

    # add times for stratify strategies
    print('fitting stratify strategies')
    for idx, x in tqdm(enumerate(strategy_list[n_base_strategies:])):
        start_time = time.time()
        x.fit(forc_xs, forc_ys, save_location = directory)
        end_time = time.time()
        fit_time_dict[strategy_names[n_base_strategies + idx]] = end_time - start_time
        
    assert len(strategy_names) == len(strategy_list)
    return strategy_list, strategy_names, fit_time_dict
    

def fit_dynamic_strategies(forc_xs, forc_ys, strategy_list, strategy_types, 
                           directory = None, sparse_dynamic_strategies = [], dense_dynamic_strategies = []):
    
    sparse_dynamics = []
    sparse_dystrat_names = []
    dense_dynamics = []
    dense_dystrat_names = []
    strategy_type_string = '_'.join(strategy_types)
    dystrat_directory = directory + f'{strategy_type_string}'
    fit_time_dict = {}
    for clf_name_pair in sparse_dynamic_strategies:
        clf, name = clf_name_pair[0], clf_name_pair[1]
        sparse_dystrat = DynamicStrategy(strategy_list, clf, sparse = True)
        start_time = time.time()
        sparse_dystrat.fit(forc_xs, forc_ys, save_location = dystrat_directory + name)
        end_time = time.time()
        fit_time_dict['clf_'+name] = end_time - start_time
        sparse_dynamics.append(sparse_dystrat)
        sparse_dystrat_names.append(name)
        
    for reg_name_pair in dense_dynamic_strategies:
        reg, name = reg_name_pair[0], reg_name_pair[1]
        dense_dystrat = DynamicStrategy(strategy_list, reg, sparse = False)
        start_time = time.time()
        dense_dystrat.fit(forc_xs, forc_ys, save_location = dystrat_directory + name)
        end_time = time.time()
        fit_time_dict['reg_'+name] = end_time - start_time
        dense_dynamics.append(dense_dystrat)
        dense_dystrat_names.append(name)
    
    return sparse_dynamics, sparse_dystrat_names, dense_dynamics, dense_dystrat_names, fit_time_dict

def eval_strategies(strategy_list, strategy_names, xs, ys, save_string, save_some_preds = True):
    predictions = []
    for idx,x in tqdm(enumerate(strategy_list)):
        # print(f'Running {strategy_names[idx]}')
        preds = x.predict(xs)
        predictions.append(preds)
    errors = [mse(preds, ys) for preds in predictions]
    error_dfs = [get_horizon_level_stats(error) for error in errors]
    stats_df = pd.concat(error_dfs, keys = strategy_names, names = ['strategy', 'horizon'])
    stats_df.to_csv(save_string)
    
    if not save_some_preds:
        return stats_df    # nothing else to do

    n_samples          = len(xs)
    sample_idx         = np.linspace(0, n_samples - 1, num=10, dtype=int)
    input_col_labels   = [f"t{j}" for j in range(xs.shape[1])]
    output_col_labels  = [f"t+{j}" for j in range(ys.shape[1])]

    
    save_string = save_string.split('stats')[0] + 'predictions/'
    try:
        os.makedirs(save_string, exist_ok=True)
        print(f"The directory '{save_string}' is ready (either it existed or was created).")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    # 3a. Inputs: one DataFrame, one row per selected sequence
    inputs_df = pd.DataFrame(
        xs[sample_idx].reshape(10, -1),
        index=sample_idx,
        columns=input_col_labels,
    )
    inputs_df.index.name = "sample"
    inputs_df.to_csv(save_string + "inputs.csv")

    # 3b. Outputs: one DataFrame per selected sequence
    for i in sample_idx:
        rows = {"ground_truth": ys[i].reshape(-1)}
        for name, preds in zip(strategy_names, predictions):
            rows[name] = preds[i].reshape(-1)

        out_df = pd.DataFrame.from_dict(rows, orient="index", columns=output_col_labels)
        out_df.index.name = "strategy"
        out_path = f"{save_string}outputs_{i}.csv"
        out_df.to_csv(out_path)

    return stats_df
        
    
    
def run_experiment(univariate_time_series, forecasting_function, H_ahead, window_size, train_p, val_p, test_p, strategy_types,
                                rel_directory = None, verbose = True, tiny = False,
                                sparse_dynamic_strategies = [], dense_dynamic_strategies = []):
    
    # get the splits
    forc_xs, forc_ys, val_xs, val_ys, test_xs, test_ys = handle_data_splits(univariate_time_series, H_ahead, window_size, train_p, val_p, test_p, verbose = verbose, tiny = tiny)
    # create the directory
    save_folder = 'torch_models/'
    if tiny :
        settings = f'tiny_h_{H_ahead}_w_{window_size}'
    else:
        settings = f'h_{H_ahead}_w_{window_size}_t_{train_p}'
    directory = save_folder + rel_directory + settings + '/'
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"The directory '{directory}' is ready (either it existed or was created).")
    except Exception as e:
        print(f"An error occurred: {e}")
    # make a dictionary to store the time taken to fit each strategy
    
    
    # fit the fixed strategies
    fixed_strategies, fixed_names, fixed_fit_time_dict = fit_fixed_strategies(forc_xs, forc_ys, forecasting_function, H_ahead, strategy_types, directory = directory, verbose = verbose)
    # fit the dynamic strategies
    print('Fitting dynamic strategies')
    sparse_strategies, sparse_names, dense_strategies, dense_names, dynamic_fit_time_dict = fit_dynamic_strategies(forc_xs, forc_ys, fixed_strategies, strategy_types, directory, sparse_dynamic_strategies, dense_dynamic_strategies)
    # fit the fixed ensemble
    average_ensemble = FixedEnsemble(fixed_strategies)
    learnt_ensemble = FixedEnsemble(fixed_strategies)
    print('fitting learnt ensemble')
    learnt_ensemble.fit(forc_xs, forc_ys)
    print('fitted')

    # save the fit times as a df
    fit_time_dict = {**fixed_fit_time_dict, **dynamic_fit_time_dict}
    
    
    # concatenate the names and strategies
    strategy_list = fixed_strategies + sparse_strategies + dense_strategies
    strategy_names = fixed_names + sparse_names + dense_names
    strategy_list.append(average_ensemble)
    strategy_names.append('average_ensemble')
    strategy_list.append(learnt_ensemble)
    strategy_names.append('learnt_ensemble')
    
    # eval the strategies
    if tiny:
        save_folder = 'tiny_stratify_rerun_results/' + rel_directory + settings + '/'
    else:
        save_folder = 'stratify_rerun_results/' + rel_directory + settings + '/'
    
    try:
        os.makedirs(save_folder, exist_ok=True)
        print(f"The directory '{save_folder}' is ready (either it existed or was created).")
    except Exception as e:
        print(f"An error occurred: {e}")

    fit_time_df = pd.DataFrame.from_dict(fit_time_dict, orient='index', columns=['fit_time'])
    fit_time_df.to_csv(save_folder + 'fit_times.csv')
    
    save_string =  save_folder + 'train_stats.csv'
    print('running evals')
    print('train stats')
    eval_strategies(strategy_list, strategy_names, forc_xs, forc_ys, save_string)
    save_string = save_folder +  'val_stats.csv'
    print('val stats')
    eval_strategies(strategy_list, strategy_names, val_xs, val_ys, save_string)
    save_string = save_folder +  'test_stats.csv'
    print('test stats')
    eval_strategies(strategy_list, strategy_names, test_xs, test_ys, save_string)
    
    print(f'results saved to {save_folder}/')