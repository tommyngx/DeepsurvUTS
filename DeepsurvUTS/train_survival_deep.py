import itertools
import pandas as pd
import numpy as np
import datetime
import json
import os

import torch # For building the networks 
import torchtuples as tt # Some useful functions
from torchtuples.callbacks import Callback
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ReduceLROnPlateau
from pycox.evaluation import EvalSurv

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"
import matplotlib.pyplot as plt
from pycox.models import LogisticHazard, PMF, DeepHitSingle, CoxPH
from pycox.models import DeepHitSingle

import itertools
    
class Concordance(tt.cb.MonitorMetrics):
    def __init__(self, x, durations, events, per_epoch=1, discrete=False, verbose=True):
        super().__init__(per_epoch)
        self.x = x
        self.durations = durations
        self.events = events
        self.verbose = verbose
        self.discrete = discrete
    
    def on_epoch_end(self):
        super().on_epoch_end()
        if self.epoch % self.per_epoch == 0:
            if not(self.discrete):
                _ = self.model.compute_baseline_hazards()
                surv = self.model.predict_surv_df(self.x)
            else:
                surv = self.model.interpolate(10).predict_surv_df(self.x)
                
            ev = EvalSurv(surv, self.durations, self.events)
            concordance = ev.concordance_td()
            self.append_score('concordance', concordance)
            
            if self.verbose:
                print('concordance:', round(concordance, 5))

def score_model(model, data, durations, events, discrete=False):
    if not(discrete):
        surv = model.predict_surv_df(data)
    else:
        surv = model.interpolate(10).predict_surv_df(data)
    return EvalSurv(surv, durations, events, censor_surv='km').concordance_td()

def train_deep_surv(train, val, test, model_obj, out_features,
                    n_nodes, n_layers, dropout, lr=0.01,
                    batch_size=16, epochs=500, output_bias=False,
                    tolerance=10,
                    model_params={}, discrete=False,
                    print_lr=True, print_logs=True, verbose=True,
                    save_path=None, load_path=None):
    """
    Train a deep survival model with options to save and load the model.

    Args:
        train, val, test: Datasets for training, validation, and testing.
        model_obj: Model class (e.g., CoxPH).
        out_features: Number of output features.
        n_nodes, n_layers, dropout: Model architecture hyperparameters.
        lr: Learning rate.
        batch_size: Batch size.
        epochs: Number of epochs.
        output_bias: Include bias in output layer.
        tolerance: Patience for early stopping.
        model_params: Additional parameters for the model.
        discrete: Use discrete survival modeling.
        print_lr: Print learning rate.
        print_logs: Plot logs if True.
        verbose: Verbosity during training.
        save_path: Path to save the trained model.
        load_path: Path to load a pre-trained model.

    Returns:
        logs_df: DataFrame of training logs.
        model: Trained model.
        scores: Train, validation, and test concordance scores.
    """
    # Prepare the network
    in_features = train[0].shape[1]
    num_nodes = [n_nodes] * n_layers
    batch_norm = True

    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features,
        batch_norm, dropout, output_bias=output_bias)

    opt = torch.optim.Adam
    model = model_obj(net, opt, **model_params)
    model.optimizer.set_lr(lr)

    # Load pre-trained model if path is provided
    if load_path and os.path.exists(load_path):
        model.load_net(load_path)
        print(f"Model loaded from {load_path}")

    callbacks = [
        tt.callbacks.EarlyStopping(patience=tolerance),
        Concordance(val[0], val[1][0], val[1][1], per_epoch=5, discrete=discrete)
    ]

    # Train the model
    log = model.fit(
        train[0], train[1], batch_size, epochs, callbacks, verbose,
        val_data=val, val_batch_size=batch_size
    )

    # Convert logs to a DataFrame
    logs_df = log.to_pandas().reset_index().melt(
        id_vars="index", value_name="loss", var_name="dataset"
    )

    # Plot training logs
    if print_logs:
        plt.figure(figsize=(10, 6))
        for dataset in logs_df['dataset'].unique():
            subset = logs_df[logs_df['dataset'] == dataset]
            plt.plot(subset['index'], subset['loss'], label=dataset)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Save the trained model if path is provided
    if save_path:
        model.save_net(save_path)
        print(f"Model saved to {save_path}")

    # Scoring the model
    scores = {
    'train': round(score_model(model, train[0], train[1][0], train[1][1]), 4),
    'val': round(score_model(model, val[0], val[1][0], val[1][1]), 4),
    'test': round(score_model(model, test[0], test[1][0], test[1][1]), 4)
    }

    return logs_df, model, scores

def train_deep_surv_ori(train, val, test, model_obj, out_features,
                    n_nodes, n_layers, dropout , lr =0.01, 
                    batch_size = 16, epochs = 500, output_bias=False,  
                    tolerance=10, 
                    model_params = {}, discrete= False,
                    print_lr=True, print_logs=True, verbose = True):
    
    in_features = train[0].shape[1]
    num_nodes = [n_nodes]*(n_layers)
    batch_norm = True
    
    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features, 
        batch_norm, dropout, output_bias=output_bias)

    opt = torch.optim.Adam
    model = model_obj(net, opt, **model_params)
    model.optimizer.set_lr(lr)

    callbacks = [
        tt.callbacks.EarlyStopping(patience=15),
        Concordance(val[0], val[1][0], val[1][1], per_epoch=5, discrete=discrete)
    ]

    log = model.fit(train[0], train[1], batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)

    logs_df = log.to_pandas().reset_index().melt(
        id_vars="index", value_name="loss", var_name="dataset").reset_index()
    
    #print("Last lr", lr_scheduler.get_last_lr())
        
    if print_logs:
        fig = px.line(logs_df, y="loss", x="index", color="dataset", width=800, height = 400)
        fig.show()
        
    # scoring the model
    scores = {
    'train': round(score_model(model, train[0], train[1][0], train[1][1]), 4),
    'val': round(score_model(model, val[0], val[1][0], val[1][1]), 4),
    'test': round(score_model(model, test[0], test[1][0], test[1][1]), 4)
    }
        
    return logs_df, model, scores

def grid_search_deep(train, val, test, out_features, grid_params, model_obj):
    best_score = -100
    
    n = 1
    for k, v in grid_params.items():
        n*=len(v)
        
    print(f'{n} total scenario to run')
    
    result = {}
    
    try: 
        for i, combi in enumerate(itertools.product(*grid_params.values())):
            params = {k:v for k,v in zip(grid_params.keys(), combi)}

            params_ = params.copy()
            if 'model_params' in params_.keys():
                params_['model_params'] = {k:v for k,v in params['model_params'].items() if k!='duration_index'}

            print(f'{i+1}/{n}: params: {params_}')

            logs_df, model, scores = train_deep_surv(train, val, test, model_obj,out_features,
                                      print_lr=False, print_logs=False, verbose = True, **params)

            result[i] = {}
            for k, v in params_.items():
                result[i][k] = v
            result[i]['lr'] = model.optimizer.param_groups[0]['lr']
            for k, score in scores.items():
                result[i]['score_'+k] = score

            score = scores['test']
            print('Current score: {} vs. best score: {}'.format(score, best_score))

            if best_score < score:
                best_score = score
                best_model = model
    
    except KeyboardInterrupt:
        pass
        
    table = pd.DataFrame.from_dict(result, orient='index')
    
    return best_model, table.sort_values(by="score_test", ascending=False).reset_index(drop=True)


def load_model(filename, path, model_obj, in_features, out_features, params):
    num_nodes = [int(params["n_nodes"])] * (int(params["n_layers"]))
    del params["n_nodes"]
    del params["n_layers"]

    if 'model_params' in params.keys():
        model_params = json.loads(params['model_params'].replace('\'', '\"'))
        del params['model_params']
        net = tt.practical.MLPVanilla(
            in_features=in_features, out_features=out_features, num_nodes=num_nodes, **params)
        model = model_obj(net, **model_params)
    else:
        net = tt.practical.MLPVanilla(
            in_features=in_features, out_features=out_features, num_nodes=num_nodes, **params)
        model = model_obj(net)
    model.load_net(os.path.join(path, filename))

    return model


import os
import numpy as np
import pandas as pd

def train_and_save_deepsurv(train_x, train_y, test_x, test_y, cols_x, save_folder, params, train_deep_surv, CoxPH, **kwargs):
    """
    Train a DeepSurv model, evaluate it, and save results to a specified folder.

    Args:
        train_x (pd.DataFrame): Training features.
        train_y (pd.DataFrame): Training target (time-to-event and censoring).
        test_x (pd.DataFrame): Testing features.
        test_y (pd.DataFrame): Testing target (time-to-event and censoring).
        cols_x (list): List of feature column names.
        save_folder (str): Path to the folder where models and results should be saved.
        params (dict): Model hyperparameters.
        train_deep_surv (function): Function to train the DeepSurv model.
        CoxPH (class): Cox proportional hazards loss function or model.
        **kwargs: Additional arguments for `train_deep_surv`.

    Returns:
        pd.DataFrame: Updated results DataFrame.
    """
    # Create folders for saving outputs
    models_folder = os.path.join(save_folder, "models")
    results_folder = os.path.join(save_folder, "results")
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    # Prepare the data
    get_target = lambda df: (df['time2event'].values, df['censored'].values)
    y_train = get_target(train_x)
    y_test = get_target(test_x)

    train = (np.array(train_x[cols_x]).astype(np.float32), y_train)
    test = (np.array(test_x[cols_x]).astype(np.float32), y_test)

    print("Training data and target prepared.")

    # Train the DeepSurv model
    logs_df, model, scores = train_deep_surv(
        train,
        test,  # No separate validation set, using test here
        test,
        CoxPH,
        out_features=1,
        **params,
        **kwargs
    )

    # Save the trained model using .save_net()
    model_path = os.path.join(models_folder, "deepsurv_DL.pt")
    model.save_net(model_path)
    print(f"Model saved to: {model_path}")

    # Prepare result entry
    result_entry = {
        'model': 'deepsurv',
        'score_train': scores['train'],
        'score_test': scores['test'],
        'n_nodes': params.get('n_nodes'),
        'n_layers': params.get('n_layers'),
        'dropout': params.get('dropout'),
        'lr': params.get('lr'),
        'model_params': params.get('model_params'),
        'batch_size': params.get('batch_size'),
        'discrete': params.get('discrete')
    }

    # Append or replace scores in result.csv
    result_csv_path = os.path.join(results_folder, "result.csv")
    new_entry = pd.DataFrame([result_entry])

    if os.path.exists(result_csv_path):
        # Load existing results
        existing_results = pd.read_csv(result_csv_path)
        existing_results = existing_results[existing_results['model'] != 'deepsurv']

        # Check if 'deepsurv_DL' already exists
        if 'deepsurv' in existing_results['model'].values:
            # Replace the existing entry
            existing_results.loc[existing_results['model'] == 'deepsurv', result_entry.keys()] = new_entry.values
        else:
            # Append the new entry
            existing_results = pd.concat([existing_results, new_entry], ignore_index=True)
    else:
        # Create a new results file if not already present
        existing_results = new_entry

    # Save the updated results
    existing_results.to_csv(result_csv_path, index=False)
    print(scores)
    print(f"Updated results saved to: {result_csv_path}")

    # Return updated results
    return existing_results


def grid_search_deep_models(train_x, train_y, test_x, test_y, cols_x, model_target, out_features, grid_params, model_obj, train_deep_surv, **kwargs):
    """
    Perform a grid search over DeepSurv or DeepHit models, evaluate them, and return the best model and results table.

    Args:
        train_x (pd.DataFrame): Training features.
        train_y (pd.DataFrame): Training target (time-to-event and censoring).
        test_x (pd.DataFrame): Testing features.
        test_y (pd.DataFrame): Testing target (time-to-event and censoring).
        cols_x (list): List of feature column names.
        col_target (str): Target column name for survival analysis.
        model_target (str): Specify the model type ('Deepsurv' or 'DeepHit').
        out_features (int): Number of output features for the model.
        grid_params (dict): Grid of parameters for the model.
        model_obj (class): DeepSurv or DeepHit model class or object.
        train_deep_surv (function): Function to train DeepSurv or DeepHit models.
        **kwargs: Additional arguments for `train_deep_surv`.

    Returns:
        tuple: The best model and a DataFrame containing all results sorted by test score.
    """
    # Data preparation based on model_target
    if model_target == 'Deepsurv':
        get_target = lambda df: (df['time2event'].values, df['censored'].values)
        y_train = get_target(train_x)
        y_test = get_target(test_x)
        train = (np.array(train_x[cols_x]).astype(np.float32), y_train)
        test = (np.array(test_x[cols_x]).astype(np.float32), y_test)
    elif model_target == 'DeepHit':
        num_durations = int(train_x['time2event'].max())
        labtrans = DeepHitSingle.label_transform(num_durations)
        get_target = lambda df: (df['time2event'].values, df['censored'].values)
        y_train = labtrans.fit_transform(*get_target(train_x))
        y_test = labtrans.transform(*get_target(test_x))
        train = (np.array(train_x[cols_x]).astype(np.float32), y_train)
        test = (np.array(test_x[cols_x]).astype(np.float32), y_test)
    else:
        raise ValueError("Invalid model_target. Must be 'Deepsurv' or 'DeepHit'.")

    # Initialize variables
    best_score = -np.inf
    n_combinations = np.prod([len(v) for v in grid_params.values()])
    print(f'{n_combinations} total scenarios to run')

    results = {}
    best_model = None

    try:
        # Iterate through all combinations of parameters
        for i, combi in enumerate(itertools.product(*grid_params.values())):
            params = {k: v for k, v in zip(grid_params.keys(), combi)}
            params_ = params.copy()

            # Handle nested model parameters if applicable
            if 'model_params' in params_.keys():
                params_['model_params'] = {k: v for k, v in params_['model_params'].items() if k != 'duration_index'}

            print(f'{i+1}/{n_combinations}: params: {params_}')

            # Train the model
            logs_df, model, scores = train_deep_surv(
                train, test, test, model_obj, out_features, print_logs= False,
                **params, **kwargs
            )

            # Collect results for this combination
            results[i] = {**params_}
            results[i]['lr'] = model.optimizer.param_groups[0]['lr'] if hasattr(model, 'optimizer') else None
            for k, score in scores.items():
                results[i][f'score_{k}'] = score

            # Compare and update the best model
            current_score = scores.get('test', -np.inf)
            print(f'Current score: {current_score} vs. Best score: {best_score}')
            if best_score < current_score:
                best_score = current_score
                best_model = model

    except KeyboardInterrupt:
        print("Grid search interrupted by user.")

    # Compile results into a DataFrame
    results_table = pd.DataFrame.from_dict(results, orient='index')
    results_table = results_table.sort_values(by='score_test', ascending=False).reset_index(drop=True)

    return best_model, results_table


def train_and_save_deephit(train_x, train_y, test_x, test_y, cols_x, save_folder, params, train_deep_surv, **kwargs):
    """
    Train a DeepHit model, evaluate it, and save results to a specified folder.

    Args:
        train_x (pd.DataFrame): Training features.
        train_y (pd.DataFrame): Training target (time-to-event and censoring).
        test_x (pd.DataFrame): Testing features.
        test_y (pd.DataFrame): Testing target (time-to-event and censoring).
        cols_x (list): List of feature column names.
        save_folder (str): Path to the folder where models and results should be saved.
        params (dict): Model hyperparameters.
        train_deep_surv (function): Function to train the DeepHit model.
        **kwargs: Additional arguments for `train_deep_surv`.

    Returns:
        pd.DataFrame: Updated results DataFrame.
    """
    # Create folders for saving outputs
    models_folder = os.path.join(save_folder, "models")
    results_folder = os.path.join(save_folder, "results")
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    # Prepare the data for DeepHit
    num_durations = int(train_x['time2event'].max())
    labtrans = DeepHitSingle.label_transform(num_durations)
    get_target = lambda df: (df['time2event'].values, df['censored'].values)

    y_train = labtrans.fit_transform(*get_target(train_x))
    y_test = labtrans.transform(*get_target(test_x))

    train = (np.array(train_x[cols_x]).astype(np.float32), y_train)
    test = (np.array(test_x[cols_x]).astype(np.float32), y_test)

    print("Training data and target prepared for DeepHit.")

    # Train the DeepHit model
    logs_df, model, scores = train_deep_surv(
        train,
        test,  # No separate validation set, using test here
        test,
        DeepHitSingle,  # Specify the DeepHit model class
        tolerance=10,
        print_lr=True, print_logs=True,
        out_features = num_durations,
        **params,
        **kwargs
    )

    # Save the trained model using .save_net()
    model_path = os.path.join(models_folder, "deephit_DL.pt")
    model.save_net(model_path)
    print(f"Model saved to: {model_path}")

    # Prepare result entry
    result_entry = {
        'model': 'deephit',
        'score_train': scores['train'],
        'score_test': scores['test'],
        'n_nodes': params.get('n_nodes'),
        'n_layers': params.get('n_layers'),
        'dropout': params.get('dropout'),
        'lr': params.get('lr'),
        'model_params': params.get('model_params'),
        'batch_size': params.get('batch_size'),
        'discrete': params.get('discrete')
    }

    # Append or replace scores in result.csv
    result_csv_path = os.path.join(results_folder, "result.csv")
    new_entry = pd.DataFrame([result_entry])

    if os.path.exists(result_csv_path):
        # Load existing results
        existing_results = pd.read_csv(result_csv_path)
        existing_results = existing_results[existing_results['model'] != 'deephit']

        # Check if 'deephit' already exists
        if 'deephit' in existing_results['model'].values:
            # Replace the existing entry
            existing_results.loc[existing_results['model'] == 'deephit', result_entry.keys()] = new_entry.values
        else:
            # Append the new entry
            existing_results = pd.concat([existing_results, new_entry], ignore_index=True)
    else:
        # Create a new results file if not already present
        existing_results = new_entry

    # Save the updated results
    existing_results.to_csv(result_csv_path, index=False)
    print(scores)
    print(f"Updated results saved to: {result_csv_path}")

    # Return updated results
    return existing_results


