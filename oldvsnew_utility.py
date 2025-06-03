import itertools
import numpy as np
from typing import Iterable, Generator

# Mapping from flavour to the corresponding coordinate index
FIRST = {'r': 0, 'd': 1, 'i': 2}
SECOND = {'r': 3, 'd': 4, 'i': 5}


# This can be used with optuna but the final code implementation is not done yet. Random sample optimsation used instead.
def generate_vectors(
    s_factors: Iterable[int],
    *,
    distinct_flavours: bool = False
) -> Generator[np.ndarray, None, None]:
    """
    Yield every admissible 6-D integer vector under the rules.

    Parameters
    ----------
    s_factors :
        Iterable of integers the coordinates are allowed to take.
        Zero may be present; it is ignored for the first slot.
    distinct_flavours :
        If True the second flavour must differ from the first;
        leave at False to allow repetition.

    Yields
    ------
    numpy.ndarray (shape: (6,))
        A feasible vector (r1,d1,i1,r2,d2,i2).
    """
    s_factors = list(s_factors)
    for f1, v1 in itertools.product(FIRST, s_factors):
        if v1 == 0:
            continue
        base = np.zeros(6, dtype=int)
        base[FIRST[f1]] = v1
        # case A – second slot all zeros
        yield base.copy()
        # case B – choose one non-zero flavour in the second slot
        for f2, v2 in itertools.product(SECOND, s_factors):
            if v2 == 0:
                continue
            if distinct_flavours and f2 == f1:
                continue
            vec = base.copy()
            vec[SECOND[f2]] = v2
            yield vec

def vector_to_strategy_name(vec, H_ahead=None):
    """
    Convert a 6-element vector [RECMO, DIRMO, DIRREC, RECMO, DIRMO, DIRREC]
    into a strategy name string.
    Examples:
        [1,0,0,0,0,0] => 'RECMO_1'
        [2,0,0,0,0,5] => 'stratifyRECMO_2_DIRREC_5'
        [0,5,0,0,0,0] => 'DIRMO_5'
        [1,0,0,0,2,0] => 'stratifyRECMO_1_DIRMO_2'
    """
    vec = np.array(vec)
    if len(vec) == 3:
        vec = np.append(vec, [0,0,0])
    names = ['RECMO', 'DIRMO', 'DIRREC']
    first = [(names[i], vec[i]) for i in range(3) if vec[i] != 0]
    second = [(names[i], vec[i+3]) for i in range(3) if vec[i+3] != 0]
    if not second:
        if H_ahead is not None and first[0][1] == H_ahead:
            first[0] = ('RECMO', H_ahead)
        return f"{first[0][0]}_{first[0][1]}"
    else:
        parts = []
        for (n, v) in first + second:
            if H_ahead is not None and v == H_ahead:
                n = 'RECMO'
            parts.append(f"{n}_{v}")
        return "stratify" + "_".join(parts)

def strategy_p(df, vector, key=None):
    if key is None:
        p = df.loc[vector_to_strategy_name(vector)]
    else:
        p = df.loc[vector_to_strategy_name(vector)][key]
    return p

def name_strategy_p(df, name, key=None):
    if key is None:
        p = df.loc[name]
    else:
        p = df.loc[name][key]
    return p

def sample_performance(val_stats, test_stats, pool, n, stat=np.min, time_df = None, weights_shared = False):
    if n > len(pool):
        n = len(pool)
    idx = np.random.choice(len(pool), n, replace=False)
    names = [pool[i] for i in idx]
    val_performances = [name_strategy_p(val_stats, name, 'mean') for name in names]
    opt = stat(val_performances)
    opt_name = names[val_performances.index(opt)]
    test_perf = name_strategy_p(test_stats, opt_name, 'mean')
    if time_df is None:
        return test_perf, None
    else:
        # sum the time of sampled strategies
        time = 0
        # if stratify is in more than one of the names
        stratify_names = [s for s in names if 'stratify' in s]
        if len(stratify_names) > 0 and stratify_names[0] != 'stratifyRECMO_1_DIRMO_1' and weights_shared:
            base_strategies = list(set(['_'.join(s.strip('stratify').split('_')[:2]) for s in stratify_names]))
            base_train_time = sum([time_df.loc[x]['fit_time'] for x in list(set(base_strategies))])
            time += base_train_time
    

        for name in names:
            time += time_df.loc[name]['fit_time']
        return test_perf, time

def simulate(n, val_stats, test_stats, old_strategies, new_strategies, R=1000, stat=np.min, time_df = None, weights_shared = False):
    old_stats, new_stats = [], []
    old_times, new_times = [], []
    for _ in range(R):
        old_perf, old_time = sample_performance(val_stats, test_stats, old_strategies, n, stat, time_df = time_df, weights_shared = weights_shared)
        new_perf, new_time = sample_performance(val_stats, test_stats, new_strategies, n, stat, time_df = time_df, weights_shared = weights_shared)
        old_stats.append(old_perf)
        new_stats.append(new_perf)
        old_times.append(old_time)
        new_times.append(new_time)
    return np.array(old_stats), np.array(new_stats), np.array(old_times), np.array(new_times)

def sample_performance_SO(val_stats_hmean, test_stats_hmean, space, n_calls, n_initial_points=1, random_state=0, verbose=False):
    # Helper functions (rank_to_value, etc.) should be defined in your script before calling this function.
    def strategy_p(df, vector, key=None):
        if key is None:
            p = df.loc[vector_to_strategy_name(vector)]
        else:
            p = df.loc[vector_to_strategy_name(vector)][key]
        return p
    val_obj = lambda vector: strategy_p(val_stats_hmean, vector, 'mean')
    test_obj = lambda vector: strategy_p(test_stats_hmean, vector, 'mean')
    if len(space) == 2:
        def map_to_vector(f1, v1_rank):
            FIRST = {'r': 0, 'd': 1, 'i': 2}
            v1 = rank_to_value[v1_rank]
            vec = np.zeros(3, dtype=int)
            vec[FIRST[f1]] = v1
            return vec
    else:
        def map_to_vector(f1, v1_rank, f2, v2_rank):
            FIRST = {'r': 0, 'd': 1, 'i': 2}
            SECOND = {'r': 3, 'd': 4, 'i': 5}
            v1 = rank_to_value[v1_rank]
            v2 = rank_to_value[v2_rank]
            vec = np.zeros(6, dtype=int)
            vec[FIRST[f1]] = v1
            if v2 != 0:
                vec[SECOND[f2]] = v2
            return vec
    @use_named_args(space)
    def objective(**params):
        vec = map_to_vector(**params)
        return val_obj(vec)
    result = forest_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        base_estimator="RF",
        random_state=random_state,
        verbose=verbose,
        n_initial_points=n_initial_points
    )
    best_params = result.x
    best_vec = map_to_vector(*best_params)
    best_score = result.fun
    test_score = test_obj(best_vec)
    if verbose:
        print("Best score :", best_score)
        print("Best 5-tuple:", best_params)
        print("Best 6-vec  :", best_vec)
        print("Test score  :", test_score)
    return test_score

def simulate_SO(n, val_stats_hmean, test_stats_hmean, old_space, new_space, R=1000, ignore_warning=True):
    if ignore_warning:
        import warnings
        warnings.filterwarnings("ignore", message="The objective has been evaluated at point")
    old_stats, new_stats = [], []
    old_times, new_times = [], []
    for _ in range(R):
        old_perf, old_time = sample_performance_SO(val_stats_hmean, test_stats_hmean, space=old_space, n_calls=n)
        new_perf, new_time = sample_performance_SO(val_stats_hmean, test_stats_hmean, space=new_space, n_calls=n)
        old_stats.append(old_perf)
        new_stats.append(new_perf)
        old_times.append(old_time)
        new_times.append(new_time)
    return np.array(old_stats), np.array(new_stats), np.array(old_times), np.array(new_times)
