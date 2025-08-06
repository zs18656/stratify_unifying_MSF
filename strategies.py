# strategies.py
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
    Stratify base--correction forecaster.

    Residuals are defined as

        e = beta - y,

    where beta is the base forecast. The rectifier predicts e_hat, and the
    corrected forecast is

        y_hat = beta - e_hat.

    For RecMO and DirRecMO rectifiers, recursive inputs remain in the original
    forecast-value space. Each predicted residual block is combined with the
    corresponding base forecast, and the resulting corrected forecast is used
    when constructing subsequent rectifier inputs.
    """

    def __init__(self, base_forecaster, residual_forecaster, rectifier, H_ahead):
        self.base_forecaster = base_forecaster
        self.residual_forecaster = residual_forecaster
        self.rectifier = rectifier
        self.H_ahead = H_ahead

        if rectifier is None or not hasattr(rectifier, "name"):
            raise ValueError("rectifier must have a .name attribute.")

        self.method = rectifier.name.lower()

        if self.method not in {"recmo", "dirrec", "dirmo"}:
            raise ValueError(
                f"Unknown rectifier.name '{self.method}'. "
                "Expected 'recmo', 'dirrec' or 'dirmo'."
            )

        if self.method in {"recmo", "dirrec"}:
            self.s_2 = rectifier.MO_size

            if H_ahead % self.s_2:
                raise ValueError(
                    f"H_ahead ({H_ahead}) must be a multiple of "
                    f"rectifier.MO_size ({self.s_2})."
                )

            self.n_blocks = H_ahead // self.s_2

            if self.method == "dirrec":
                self.block_models = [
                    deepcopy(self.rectifier.function_family)
                    for _ in range(self.n_blocks)
                ]
        else:
            self.s_2 = None
            self.n_blocks = 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_2d(values):
        """Ensure predictions have shape (n_samples, n_outputs)."""
        values = np.asarray(values)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return values

    def _fit_or_load_rectifier_model(
        self,
        model,
        X,
        target,
        file,
        family_name,
    ):
        """
        Fit or load a rectifier model.

        This helper handles persistence only; recursive state construction is
        defined separately by the corresponding strategy implementation.
        """
        target = self._as_2d(target)
        fit_target = target.ravel() if target.shape[1] == 1 else target

        neural_names = {"MLP", "RNN", "LSTM", "Transformer"}

        if file:
            try:
                if family_name in neural_names:
                    model.fit(X, fit_target, init_only=True)
                    model.load_state_dict(torch.load(f"{file}.pth"))
                    print(f"loaded pretrained {file}.pth")
                    return model

                elif family_name == "RF":
                    model = load(f"{file}.joblib")
                    print(f"loaded pretrained {file}.joblib")
                    return model

                elif family_name == "XGB":
                    model.load_model(f"{file}.json")
                    print(f"loaded pretrained {file}.json")
                    return model

                else:
                    raise ValueError(
                        f"No loading rule for model family '{family_name}'."
                    )

            except Exception as e:
                print(e)
                print(f"ERROR FINDING {file} - FITTING NEW MODEL")

        model.fit(X, fit_target)

        if file:
            if family_name in neural_names:
                torch.save(model.state_dict(), f"{file}.pth")

            elif family_name == "RF":
                dump(model, f"{file}.joblib", compress=("gzip", 3))

            elif family_name == "XGB":
                model.save_model(f"{file}.json")

        return model

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X, y, save_location=""):
        """Fit the base forecaster and residual-correction construction."""
        y = self._as_2d(y)

        # 1. Fit the base forecasting strategy.
        self.base_forecaster.fit(
            X,
            y,
            save_location=save_location,
        )

        base_preds = self._as_2d(
            self.base_forecaster.predict(X)
        )

        # 2. Construct residual targets:
        #
        #     e = beta - y
        residuals = base_preds - y

        residual_str = (
            self.base_forecaster.name
            + str(self.base_forecaster.MO_size)
            + "_residual_"
        )

        self.residual_forecaster.fit(
            X,
            residuals,
            save_location=save_location + residual_str,
        )

        # 3. Fit the selected rectifier strategy.
        rectify_str = (
            self.base_forecaster.name
            + str(self.base_forecaster.MO_size)
            + "_rectifier_"
            + self.method
            + str(self.rectifier.MO_size)
        )

        if self.method == "recmo":
            self._fit_recmo(
                X,
                residuals,
                save_location=save_location + rectify_str,
            )

        elif self.method == "dirrec":
            self._fit_dirrec(
                X,
                residuals,
                base_preds,
                save_location=save_location + rectify_str,
            )

        # For DirMO, the residual forecaster fitted above directly supplies
        # the correction over the complete forecast horizon.
        return self

    # ------------------------------------------------------------------
    # RecMO rectifier fitting
    # ------------------------------------------------------------------

    def _fit_recmo(self, X, residuals, save_location=""):
        """
        Fit one s_2-output correction model on the first residual block.

        At prediction time, the same model is applied recursively to
        fixed-width windows containing previously corrected forecasts.
        """
        target_block = residuals[:, : self.s_2]

        family_name = self.rectifier.function_family.name

        self.rectifier.model = self._fit_or_load_rectifier_model(
            model=self.rectifier.model,
            X=X,
            target=target_block,
            file=save_location,
            family_name=family_name,
        )

    # ------------------------------------------------------------------
    # DirRecMO rectifier fitting
    # ------------------------------------------------------------------

    def _fit_dirrec(self, X, residuals, base_preds, save_location=""):
        """
        Sequentially fit the DirRecMO correction models.

        Each block model predicts a residual from the accumulated forecast
        state. The residual prediction is combined with the corresponding base
        forecast, and the corrected block is appended to the state used by
        subsequent models.
        """
        rect_inputs = np.asarray(X)

        family_name = self.rectifier.function_family.name

        for block in range(self.n_blocks):
            start = block * self.s_2
            stop = (block + 1) * self.s_2

            target_block = residuals[:, start:stop]

            file = f"{save_location}_id{block}"

            self.block_models[block] = self._fit_or_load_rectifier_model(
                model=self.block_models[block],
                X=rect_inputs,
                target=target_block,
                file=file,
                family_name=family_name,
            )

            predicted_error = self._as_2d(
                self.block_models[block].predict(rect_inputs)
            )

            base_block = base_preds[:, start:stop]

            # Convert the predicted residual to a corrected forecast.
            corrected_block = base_block - predicted_error

            # Append the corrected block to the state for later models.
            rect_inputs = np.concatenate(
                [rect_inputs, corrected_block],
                axis=1,
            )

    # ------------------------------------------------------------------
    # Forecast helpers
    # ------------------------------------------------------------------

    def get_base_forecast(self, X):
        return self._as_2d(
            self.base_forecaster.predict(X)
        )

    def get_residual_forecast(self, X):
        """
        Apply the separately fitted residual forecaster to the base forecast.

        With e = beta - y, the corrected forecast is

            y_hat = beta - e_hat.
        """
        base_f = self.get_base_forecast(X)

        error_f = self._as_2d(
            self.residual_forecaster.predict(X)
        )

        return base_f - error_f

    def get_rectified_forecast(self, X):
        if self.method == "dirmo":
            return self.get_residual_forecast(X)

        elif self.method == "recmo":
            return self._rectified_recmo(X)

        else:  # dirrec
            return self._rectified_dirrec(X)

    # ------------------------------------------------------------------
    # RecMO rectifier rollout
    # ------------------------------------------------------------------

    def _rectified_recmo(self, X):
        """
        Generate a closed-loop RecMO rectified forecast.

        Each predicted residual block is combined with the corresponding base
        forecast before the corrected block is inserted into the rolling
        forecast state.
        """
        X = np.asarray(X)

        n_samples, window_size = X.shape
        base_f = self.get_base_forecast(X)

        # Rolling state: observed history followed by future forecast slots.
        rollout_state = np.concatenate(
            [
                X,
                np.zeros((n_samples, self.H_ahead)),
            ],
            axis=1,
        )

        for block in range(self.n_blocks):
            start = block * self.s_2
            stop = (block + 1) * self.s_2

            # Fixed-width RecMO input window.
            input_window = rollout_state[
                :,
                start : start + window_size,
            ]

            predicted_error = self._as_2d(
                self.rectifier.model.predict(input_window)
            )

            base_block = base_f[:, start:stop]

            # Convert the predicted residual to a corrected forecast.
            corrected_block = base_block - predicted_error

            # Insert the corrected block into the rolling forecast state.
            rollout_state[
                :,
                window_size + start : window_size + stop,
            ] = corrected_block

        return rollout_state[:, -self.H_ahead:]

    # ------------------------------------------------------------------
    # DirRecMO rectifier rollout
    # ------------------------------------------------------------------

    def _rectified_dirrec(self, X):
        """
        Generate a closed-loop DirRecMO rectified forecast.

        Each block model predicts a residual from the current accumulated
        forecast state. The corrected forecast block is appended before the
        next block model is evaluated.
        """
        X = np.asarray(X)

        base_f = self.get_base_forecast(X)

        rectified = np.zeros_like(base_f)

        # The first model receives only the observed history.
        rect_inputs = X

        for block in range(self.n_blocks):
            start = block * self.s_2
            stop = (block + 1) * self.s_2

            predicted_error = self._as_2d(
                self.block_models[block].predict(rect_inputs)
            )

            base_block = base_f[:, start:stop]

            # Convert the predicted residual to a corrected forecast.
            corrected_block = base_block - predicted_error

            rectified[:, start:stop] = corrected_block

            # Append the corrected block for use by subsequent models.
            rect_inputs = np.concatenate(
                [rect_inputs, corrected_block],
                axis=1,
            )

        return rectified

    # ------------------------------------------------------------------
    #  prediction interface
    # ------------------------------------------------------------------

    def predict(self, X, decompose=False):
        if not decompose:
            return self.get_rectified_forecast(X)

        residual_corrected = self.get_residual_forecast(X)

        rectified = (
            residual_corrected
            if self.method == "dirmo"
            else self.get_rectified_forecast(X)
        )

        return {
            "base": self.get_base_forecast(X),
            "residual": residual_corrected,
            "rectified": rectified,
        }

    # ------------------------------------------------------------------

    def __repr__(self):
        method = self.method.upper()

        return (
            f"Stratify(method={method}, H_ahead={self.H_ahead}, "
            f"base={self.base_forecaster.__class__.__name__}, "
            f"residual={self.residual_forecaster.__class__.__name__}, "
            f"rectifier="
            f"{getattr(self.rectifier, '__class__', type(None)).__name__})"
        )