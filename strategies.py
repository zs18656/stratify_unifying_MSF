from TS_functions import shiftData, moving_window
from sklearn.metrics import mean_squared_error
import numpy as np
from copy import deepcopy
import pandas as pd
from scipy.stats import rankdata
from tqdm import tqdm
import torch
import time
from joblib import dump, load
import pickle

def mse(preds, ys):
        return (preds - ys)**2

class RECMO():
    def __init__(self, function_family, H_ahead, s_parameter):
        assert H_ahead%s_parameter == 0, 'select s such that s divides H_ahead'
        
        self.function_family = function_family
        self.no_recursions = H_ahead//s_parameter
        self.MO_size = s_parameter
        self.H_ahead = H_ahead
        self.model = self.function_family
        self.name = 'recmo'
    
    def fit(self, windowed_data, ys, save_location = ''):
        
        xs, ys = windowed_data, ys
        
        file = f'{save_location}recmo{self.MO_size}'
        print(f'fitting {file}')

        try: # first try to load a model if it exists    
            if self.function_family.name in ['MLP', 'RNN', 'LSTM', 'Transformer']:
                dys = ys[:, : self.MO_size]
                self.model.fit(xs, dys, init_only = True) if self.MO_size != 1 else self.model.fit(xs, dys.ravel(), init_only = True)
            
                self.model.load_state_dict(torch.load(f'{file}.pth'))
                print(f'loaded from {file} .pth')
                
            elif self.function_family.name == 'RF':
                self.model = load(file)
                print(f'loaded from {file} .joblib')
                
            elif self.function_family.name == 'XGB':
                self.model.load_model(f'{file}.json')
                print(f'loaded from {file} .json')
                
            else:
                raise ValueError(f'No pretrained model found at {file}')
            
        except Exception as e:
            print(e)
            print(f'ERROR FINDING {file} - FITTING NEW MODEL')
            # quit()
            dys = ys[:, : self.MO_size]
            self.model.fit(xs, dys) if self.MO_size != 1 else self.model.fit(xs, dys.ravel())
            
            if len(save_location) > 0:
                if self.function_family.name in ['MLP', 'RNN', 'LSTM', 'Transformer']:
                    torch.save(self.model.state_dict(), f'{file}.pth')
                elif self.function_family.name == 'RF':
                    dump(self.model, f'{file}.joblib', compress=('gzip', 3))  # Compression level 3 is a good balance
                elif self.function_family.name == 'XGB':
                    self.model.save_model(f'{file}.json')

    def predict(self, windowed_data):
        if self.no_recursions == 1:
            return self.model.predict(windowed_data)
        
        preds = np.concatenate([windowed_data, np.zeros((windowed_data.shape[0], self.H_ahead))], axis = 1)
        window_size = len(windowed_data[0])
        for recursion_id in range(self.no_recursions):
            input_window = preds[:, recursion_id*self.MO_size: recursion_id*self.MO_size + window_size]
            # print(f'input_window shape: {input_window.shape}_{recursion_id}')
            preds_i = self.model.predict(input_window)
            if self.MO_size == 1:
                preds_i = preds_i.reshape(-1,1)
            preds[:, window_size + recursion_id*self.MO_size: window_size + (recursion_id+1)*self.MO_size] = preds_i
        preds = preds[:, -self.H_ahead:]
        return preds

    def evaluate(self, windowed_data, metric = mean_squared_error):
        xs, ys = shiftData(windowed_data[:-1], self.H_ahead)
        pred_ys = self.predict(xs)
        return np.array([metric(pred_ys[i], ys[i]) for i in range(len(pred_ys))])
    


class DIRMO():
    def __init__(self, function_family, H_ahead, s_parameter):
        assert H_ahead%s_parameter == 0, 'select s such that s divides H_ahead'
        
        self.function_family = function_family
        self.no_funcs = H_ahead//s_parameter
        self.MO_size = s_parameter
        self.H_ahead = H_ahead
        self.models = [deepcopy(self.function_family) for func in range(self.no_funcs)]
        self.name = 'dirmo'

    def fit(self, windowed_data, ys, save_location = ''):
        
        xs, ys = windowed_data, ys
            
            
        for func_id, func in enumerate(self.models):
            file = f'{save_location}dirmo{self.MO_size}_id{func_id}'
            try: # first try to load a model if it exists
                if self.function_family.name in ['MLP', 'RNN', 'LSTM', 'Transformer']:
                    dys = ys[:, self.MO_size*(func_id): self.MO_size*(func_id + 1)]
                    func.fit(xs, dys, init_only = True) if self.MO_size != 1 else func.fit(xs, dys.ravel(), init_only = True)
                    
                    func.load_state_dict(torch.load(f'{file}.pth'))
                    print(f'loaded from {file} .pth')
                elif self.function_family.name == 'RF':
                    func = load(f'{file}.joblib')
                    print(f'loaded from {file} .joblib')
                elif self.function_family.name == 'XGB':
                    func.load_model(f'{file}.json')
                    print(f'loaded from {file} .json')
                else:
                    raise ValueError(f'No pretrained model found at {file}')
                
            except:  
                dys = ys[:, self.MO_size*(func_id): self.MO_size*(func_id + 1)]
                func.fit(xs, dys) if self.MO_size != 1 else func.fit(xs, dys.ravel())
                
                if len(save_location) > 0: # save the model
                    if self.function_family.name in ['MLP', 'RNN', 'LSTM', 'Transformer']:
                        torch.save(func.state_dict(), f'{file}.pth')
                    elif self.function_family.name == 'RF':
                        dump(func, f'{file}.joblib', compress=('gzip', 3))
                    elif self.function_family.name == 'XGB':
                        func.save_model(f'{file}.json')
                        
                    

    def predict(self, windowed_data):
        preds = np.zeros([windowed_data.shape[0], self.H_ahead])

        for func_id, func in enumerate(self.models):
            preds_i = func.predict(windowed_data)
            if self.MO_size == 1:
                preds_i = preds_i.reshape(-1,1)
            preds[:, self.MO_size*(func_id): self.MO_size*(func_id + 1)] = preds_i
        
        return preds

    def evaluate(self, windowed_data, metric = mean_squared_error):
        xs, ys = shiftData(windowed_data[:-1], self.H_ahead)
        pred_ys = self.predict(xs)
        return np.array([metric(pred_ys[i], ys[i]) for i in range(len(pred_ys))])

class DIRREC():

    def __init__(self, function_family, H_ahead, s_parameter):
        assert H_ahead%s_parameter == 0, 'select s such that s divides H_ahead'
        
        self.function_family = function_family
        self.no_funcs = H_ahead//s_parameter
        self.MO_size = s_parameter
        self.H_ahead = H_ahead
        self.models = [deepcopy(self.function_family) for func in range(self.no_funcs)]
        self.name = 'dirrec'

    def fit(self, windowed_data, ys, save_location = ''):
        
        xs, ys = windowed_data, ys
            

        for func_id, func in enumerate(self.models):
            file = f'{save_location}dirrec{self.MO_size}_id{func_id}'
            try:  # first try to load a model if it exists
                    
                if self.function_family.name in ['MLP', 'RNN', 'LSTM', 'Transformer']:
                    dys = ys[:, self.MO_size*(func_id): self.MO_size*(func_id + 1)]
                    func.fit(xs, dys, init_only = True) if self.MO_size != 1 else func.fit(xs, dys.ravel(), init_only = True)
                    func.load_state_dict(torch.load(f'{file}.pth'))
                    print(f'loaded pretrained {save_location}dirrec{self.MO_size}_id{func_id}.pth')
                    xs = np.random.rand(xs.shape[0], xs.shape[1] + self.MO_size)  # add the MO_size to the input to load the next model
                    
                elif self.function_family.name == 'RF':
                    func = load(f'{file}.joblib')
                    print(f'loaded pretrained {file}.joblib')
                elif self.function_family.name == 'XGB':
                    func.load_model(f'{file}.json')
                    print(f'loaded pretrained {file}.json')
                else:
                    raise ValueError(f'No pretrained model found at {file}')
                
            except:
                dys = ys[:, self.MO_size*(func_id): self.MO_size*(func_id + 1)]
                func.fit(xs, dys) if self.MO_size != 1 else func.fit(xs, dys.ravel())
                func_pred = func.predict(xs) if self.MO_size != 1 else func.predict(xs).reshape(-1,1)
                xs = np.concatenate([xs, func_pred], axis = 1) # add the MO_size to the input to train the next model
            
                if len(save_location) > 0:
                    if self.function_family.name in ['MLP', 'RNN', 'LSTM', 'Transformer']:
                        torch.save(func.state_dict(), f'{file}.pth')
                    elif self.function_family.name == 'RF':
                        dump(func, f'{file}.joblib', compress=('gzip', 3))
                    elif self.function_family.name == 'XGB':
                        func.save_model(f'{file}.json')
                    
                        
    def predict(self, windowed_data):
        preds = np.zeros([windowed_data.shape[0], self.H_ahead])
        for func_id, func in enumerate(self.models):
            preds_i = func.predict(windowed_data) if self.MO_size != 1 else func.predict(windowed_data).reshape(-1,1)
            windowed_data = np.concatenate([windowed_data, preds_i], axis = 1)
            if self.MO_size == 1:
                preds[:, func_id] = preds_i.reshape(-1)
            else: 
                preds[:, self.MO_size*(func_id): self.MO_size*(func_id + 1)] = preds_i
        
        return preds

    def evaluate(self, windowed_data, metric = mean_squared_error):
        xs, ys = shiftData(windowed_data[:-1], self.H_ahead)
        pred_ys = self.predict(xs)
        return np.array([metric(pred_ys[i], ys[i]) for i in range(len(pred_ys))])
        

class FixedEnsemble():
    def __init__(self, strategy_list):
        self.strategy_list = strategy_list
        self.weights = np.ones(len(strategy_list))/len(strategy_list)
        
    def fit(self, xs, ys):
        preds = np.array([self.strategy_list[strat_id].predict(xs) for strat_id in range(len(self.strategy_list))])
        preds_by_strat = preds.reshape(preds.shape[0], -1)
        if ys is not None:
            ys = ys.reshape(-1)
            weights = np.linalg.lstsq(preds_by_strat.T, ys, rcond=None)[0]
            self.weights = weights
        
    def predict(self, xs):
        # preds = np.array([self.strategy_list[strat_id].predict(xs) for strat_id in range(len(self.strategy_list))])
        preds = []
        for idx in range(len(self.strategy_list)):
            # print(f'predicting with {self.strategy_list[idx]}_{idx}')
            preds.append(self.strategy_list[idx].predict(xs))
        preds = np.array(preds)
        return np.array([weight * preds[idx] for idx, weight in enumerate(self.weights)]).sum(axis = 0)
    
from sklearn.metrics import mean_squared_error, accuracy_score

class DynamicStrategy():
    def __init__(self, strategy_list, learner, sparse):
        self.strategy_list = strategy_list 
        self.learner = learner
        self.sparse = sparse
        
    def get_weights_per_instance(self, xs, trues):
        all_preds = np.array([x.predict(xs) for x in (self.strategy_list)])
        weights = []
        for instance_idx in tqdm(range(all_preds.shape[1])):
            target = trues[instance_idx]
            basis = all_preds[:, instance_idx].T
        
            if self.sparse:
                errors = [mse(target, base).mean() for base in basis.T]
                ratios = np.zeros_like(errors)
                ratios[np.argmin(errors)] = 1
            else:
                inverse_matrix = np.linalg.pinv(basis)
                ratios = np.dot(inverse_matrix, target)
                
            weights.append(ratios)
        return np.array(weights)

    def get_ensemble_predictions(self, all_preds, pred_weights):
        predictions = []
        for instance_idx in tqdm(range(all_preds.shape[1])):
            basis = all_preds[:, instance_idx].T
            prediction = np.dot(basis, pred_weights[instance_idx])
            predictions.append(prediction)
        return np.array(predictions)
    
    def fit(self, xs, ys, verbose=False, save_location = ''):
        if self.sparse:
            sparse_str = 'sparse'
        else:
            sparse_str = 'dense'
        try:
            
            try:
                self.learner.load_state_dict(torch.load(f'{save_location}_{sparse_str}_dystrat.pth'))
                print(f'loaded pretrained {save_location}.pth')
            except:
                with open(f'{save_location}_{sparse_str}_dystrat.pkl', 'rb') as file:
                    self.learner = pickle.load(file)
                print(f'loaded pretrained {save_location}_{sparse_str}_dystrat.pkl')
        except:
            train_weights = self.get_weights_per_instance(xs, ys)
            if self.sparse:
                try:
                    self.learner.fit(xs, train_weights)
                    self.multi_output = True
                except:
                    self.learner.fit(xs, train_weights.argmax(axis=0))
                    self.multi_output = False
            else:
                self.learner.fit(xs, train_weights)
                
            if len(save_location) > 0:
                try:
                    torch.save(self.learner.state_dict(), f'{save_location}.pth')
                except:
                    with open(f'{save_location}_{sparse_str}_dystrat.pkl', 'wb') as file:
                        pickle.dump(self.learner, file)
        if verbose:
            pred_weights = self.learner.predict(xs)
            if self.sparse:
                try:
                    performance = 1- accuracy_score(pred_weights, train_weights)
                except:
                    print(f"Error shape: {pred_weights.shape}")
                    print(f"True shape: {train_weights.shape}")
                    performance = 1- accuracy_score(pred_weights.argmax(axis=1), train_weights.argmax(axis=1))
            else:
                performance = mean_squared_error(pred_weights, train_weights)
                print(f"Error shape: {performance.shape}")
            print(f"Error: {performance}")
        
    def predict(self, xs, eval = False, ys = None):
        predicted_weights = self.learner.predict(xs)
        all_preds = np.array([x.predict(xs) for x in (self.strategy_list)])
        if self.sparse:
            # if self.multi_output:
            if True: # always multi output for now
                predicted_weights = predicted_weights.argmax(axis=1)
            predicted_weights = np.eye(len(self.strategy_list))[predicted_weights]

        if eval:
            assert ys is not None , "Need to provide true values for evaluation"
            optimal_weights = self.get_weights_per_instance(xs, ys)
            if self.sparse:
                try:
                    performance = 1- accuracy_score(predicted_weights, optimal_weights)
                except:
                    performance = 1- accuracy_score(predicted_weights.argmax(axis=1), optimal_weights.argmax(axis=1))
                print(f'Sparse weight 1 - accuracy: {performance}')
            else:
                performance = mean_squared_error(predicted_weights, optimal_weights)
                print(f"Dense Weigth MSE : {performance}")
            return self.get_ensemble_predictions(all_preds, predicted_weights), performance
                
        return self.get_ensemble_predictions(all_preds, predicted_weights)
    
    
import numpy as np
from copy import deepcopy

class Stratify:
    """
    A unified composite forecaster that wraps a base forecaster, a residual
    forecaster and (optionally) a rectifier.  The concrete training / inference
    strategy is chosen at runtime via ``rectifier.name``:

    * ``'recmo'`` – *Recursive Multi‑Output* (vectorised block rectification).
    * ``'dirrec'`` – *Direct‑Recursive Multi‑Output* (separate rectifier per
      block).
    * ``'dirmo'`` – *Direct Multi‑Output* (no rectification – residual model is
      the final output).

    Parameters
    ----------
    base_forecaster : object
        Model implementing ``fit(X, y)`` and ``predict(X)``.  Must expose a
        ``window_size`` attribute.
    residual_forecaster : object
        Model of the forecast residuals implementing ``fit`` / ``predict``.
    rectifier : object or None
        Model implementing ``predict`` (and commonly ``fit``) with attributes:
            * ``name``   – string in {"recmo", "dirrec", "dirmo"}
            * ``MO_size`` – block length (only required for *recmo* / *dirrec*)
            * ``function_family`` – prototype estimator to be deep‑copied for
              each block (only required for *dirrec*).
        For the *dirmo* strategy ``rectifier`` may be ``None``.
    H_ahead : int
        Forecast horizon.  For block approaches it must be an integer multiple
        of ``rectifier.MO_size``.
    """

    def __init__(self, base_forecaster, residual_forecaster, rectifier, H_ahead):
        self.base_forecaster = base_forecaster
        self.residual_forecaster = residual_forecaster
        self.rectifier = rectifier
        # correct the horizon on the rectifier
        self.rectifier.H_ahead = self.rectifier.MO_size
        self.H_ahead = H_ahead

        # Normalise strategy label
        if rectifier is None or not hasattr(rectifier, "name"):
            raise ValueError("rectifier must have a .name attribute.")

        self.method = rectifier.name.lower()

        if self.method not in {"recmo", "dirrec", "dirmo"}:
            raise ValueError(f"Unknown rectifier.name '{self.method}'. Expected 'recmo', 'dirrec' or 'dirmo'.")


        # Validate block settings for the two block‑based methods
        if self.method in {"recmo", "dirrec"}:
            self.s_2 = rectifier.MO_size
            if H_ahead % self.s_2:
                raise ValueError(f"H_ahead ({H_ahead}) must be a multiple of rectifier.MO_size ({self.s_2}).")
            self.n_blocks = H_ahead // self.s_2

            if self.method == "dirrec":
                # Prepare an independent rectifier per block using the provided prototype
                self.block_models = [deepcopy(self.rectifier.function_family) for _ in range(self.n_blocks)]
        else:
            self.s_2 = None  # not used
            self.n_blocks = 1

    # ---------------------------------------------------------------------
    # Fitting
    # ---------------------------------------------------------------------
    def fit(self, X, y, save_location = ''):
        """Fit all underlying components according to the chosen strategy."""
        # 1. Base model -----------------------------------------------------
        self.base_forecaster.fit(X, y, save_location = save_location) # base model can be loaded directly
        base_preds = self.base_forecaster.predict(X)

        # 2. Residual model --------------------------------------------------
        residual_str = self.base_forecaster.name + str(self.base_forecaster.MO_size) + '_residual_'
        residuals = base_preds - y
        self.residual_forecaster.fit(X, residuals, save_location = save_location + residual_str)
        residual_preds = np.subtract(base_preds, self.residual_forecaster.predict(X))

        # 3. Rectifier‑specific training ------------------------------------
        rectify_str = self.base_forecaster.name + str(self.base_forecaster.MO_size) + '_rectifier_' + self.method + f'{self.rectifier.MO_size}'
        if self.method == "recmo":
            self._fit_recmo(X, residuals, save_location = save_location + rectify_str)
            # print(f"rectifier {self.rectifier.name} fitted, X shae {X.shape}, residuals shape {residuals.shape}")
            final_train_error = np.mean((self.get_rectified_forecast(X) - y) ** 2)

        elif self.method == "dirrec":
            self._fit_dirrec(X, residuals, base_preds, save_location = save_location + rectify_str)
            final_train_error = np.mean((self.get_rectified_forecast(X) - y) ** 2)

        else:  # dirmo – no rectifier
            final_train_error = np.mean((residual_preds - y) ** 2)

        # Diagnostics (quick and dirty – keep or remove to taste)
        # print("--------------------------")
        # print(f"Base train error:      {np.mean((base_preds - y) ** 2):.4f}")
        # print(f"Residual train error:  {np.mean((residual_preds - y) ** 2):.4f}")
        # if self.method != "dirmo":
        #     print(f"Rectified train error: {final_train_error:.4f}")
        # print("--------------------------")
        return self

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _fit_recmo(self, X, residuals, save_location = ''):
        """Fit single rectifier on the first block of residuals (RecMO)."""
        # rectifier_targets = residuals[:, : self.s_2]
        # self.rectifier.fit(X, rectifier_targets)
        # try and load the model
        try:
            # iniitalise the weights
            dys = residuals[:, : self.s_2]
            self.model.fit(X, dys, init_only = True) if self.MO_size != 1 else self.model.fit(X, dys.ravel(), init_only = True)
            
            self.rectifier.model.load_state_dict(torch.load(f'{save_location}.pth'))
            print(f'loaded pretrained {save_location}.pth')
        except:
            print(f'ERROR FINDING {save_location} - FITTING NEW MODEL')
            # quit()
            self.rectifier.model.fit(X, residuals[:, : self.s_2])
            if len(save_location) > 0:
                torch.save(self.rectifier.model.state_dict(), f'{save_location}.pth')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _fit_dirrec(self, X, residuals, base_preds, save_location = ''):
        """Fit an independent rectifier for each block (DirRecMO)."""
        window_size = X.shape[1]
        full_series = np.concatenate([X, base_preds], axis=1)  # (n, window + H)
        # try and load the model
        for block in range(self.n_blocks):
            end = window_size + (block + 1) * self.s_2
            rect_inputs = full_series[:, : end]
            target_block = residuals[:, block * self.s_2 : (block + 1) * self.s_2]
            file = f'{save_location}_id{block}'
            try:
                if self.rectifier.function_family.name in ['MLP', 'RNN', 'LSTM', 'Transformer']:
                    dys = target_block
                    self.block_models[block].fit(rect_inputs, dys, init_only = True) if self.s_2 != 1 else self.block_models[block].fit(rect_inputs, dys.ravel(), init_only = True)
                    self.block_models[block].load_state_dict(torch.load(f'{file}.pth'))
                    print(f'loaded pretrained {file}.pth')
                else:
                    raise ValueError(f'No pretrained model found at {file}')
            except Exception as e:
                print(e)
                print(f'ERROR FINDING {file} - FITTING NEW MODEL')
                # quit()
                self.block_models[block].fit(rect_inputs, target_block)
                if len(save_location) > 0:
                    if self.rectifier.function_family.name in ['MLP', 'RNN', 'LSTM', 'Transformer']:
                        torch.save(self.block_models[block].state_dict(), f'{file}.pth')
            
        
        

    # ---------------------------------------------------------------------
    # Forecast helpers
    # ---------------------------------------------------------------------
    def get_base_forecast(self, X):
        return self.base_forecaster.predict(X)

    def get_residual_forecast(self, X):
        base_f = self.get_base_forecast(X)
        res_f = self.residual_forecaster.predict(X)
        return np.subtract(base_f, res_f)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_rectified_forecast(self, X):
        if self.method == "dirmo":
            # In direct multi‑output the rectified output is just the residual model
            return self.get_residual_forecast(X)
        elif self.method == "recmo":
            return self._rectified_recmo(X)
        else:  # dirrec
            return self._rectified_dirrec(X)

    # .....................................................................
    def _rectified_recmo(self, X):
        n_samples, window_size = X.shape
        base_f = self.get_base_forecast(X)
        full_series = np.concatenate([X, base_f], axis=1)

        # Build all rectifier inputs in one batch (vectorised formulation)
        rect_inputs = np.stack(
            [full_series[:, b * self.s_2 : window_size + b * self.s_2] for b in range(self.n_blocks)],
            axis=1,
        )  # (n, blocks, window)
        batch_inputs = rect_inputs.reshape(-1, window_size)
        batch_outputs = self.rectifier.model.predict(batch_inputs)  # (n*blocks, s2)
        rect_outputs = batch_outputs.reshape(n_samples, self.n_blocks, self.s_2)
        

        base_blocks = base_f.reshape(n_samples, self.n_blocks, self.s_2)
        corrected_blocks = base_blocks - rect_outputs
        return corrected_blocks.reshape(n_samples, self.H_ahead)

    # .....................................................................
    def _rectified_dirrec(self, X):
        n_samples, window_size = X.shape
        base_f = self.get_base_forecast(X)
        full_series = np.concatenate([X, base_f], axis=1)
        rectified = np.zeros_like(base_f)

        for block in range(self.n_blocks):
            end = window_size + (block + 1) * self.s_2
            rect_inputs = full_series[:, : end]
            block_residual_pred = self.block_models[block].predict(rect_inputs)
            rectified[:, block * self.s_2 : (block + 1) * self.s_2] = block_residual_pred

        return np.subtract(base_f, rectified)

    # ---------------------------------------------------------------------
    # Public prediction interface
    # ---------------------------------------------------------------------
    def predict(self, X, decompose = False):
        if not decompose:
            return self.get_rectified_forecast(X)
        else:
            residual = self.get_residual_forecast(X)
            rectified = residual if self.method == "dirmo" else self.get_rectified_forecast(X)
            return {
                "base": self.get_base_forecast(X),
                "residual": residual,
                "rectified": rectified,
            }

    # ---------------------------------------------------------------------
    def __repr__(self):
        method = self.method.upper()
        return (
            f"Stratify(method={method}, H_ahead={self.H_ahead}, "
            f"base={self.base_forecaster.__class__.__name__}, "
            f"residual={self.residual_forecaster.__class__.__name__}, "
            f"rectifier={getattr(self.rectifier, '__class__', type(None)).__name__})"
        )


# _____________________________________________________________________ 
# Legacy code
# _____________________________________________________________________
class STRATIFY():

    def __init__(self, base_forcaster, residual_forecaster):
        self.base_forcaster = base_forcaster
        self.residual_forecaster = residual_forecaster
        assert False, 'Updated to use Stratify class'
        
    def fit(self, windowed_data, ys, save_location = ''):
        
        xs, ys = windowed_data, ys
        try:
            base_preds = self.base_forcaster.predict(xs)
        except:
            print('failiure in base, fit base first and use .predict class method') 
            base_preds = self.base_forcaster.predict(xs)
        
        errors = np.subtract(base_preds, ys)
        
        self.residual_forecaster.fit(xs, errors, save_location = save_location)
                        
    def predict(self, windowed_data):
        base_preds = self.base_forcaster.predict(windowed_data) 
        residual_preds = self.residual_forecaster.predict(windowed_data)
        preds = base_preds - residual_preds
        return preds

    def evaluate(self, windowed_data, metric = mean_squared_error):
        xs, ys = shiftData(windowed_data[:-1], self.H_ahead)
        pred_ys = self.predict(xs)
        return np.array([metric(pred_ys[i], ys[i]) for i in range(len(pred_ys))])