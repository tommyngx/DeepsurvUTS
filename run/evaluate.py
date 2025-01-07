import os
import pandas as pd
import numpy as np
import torch
from pycox.evaluation import EvalSurv
from sksurv.metrics import integrated_brier_score, brier_score
from sksurv.functions import StepFunction
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.svm import FastSurvivalSVM
import json
import torchtuples as tt
from pycox.models import DeepHitSingle, CoxPH
import pickle

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
    
    # Load the model with weights_only=True and map to CPU if CUDA is not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_net(os.path.join(path, filename), weights_only=True, map_location=device)

    return model

def load_models_and_results(path_dir, cols_x, models_dl=['deepsurv', 'deephit']):
    """
    Load machine learning and deep learning models, and the result table.

    Args:
        path_dir (str): Base directory where models and results are stored.
        cols_x (list): List of feature column names.
        models_dl (list): Names of deep learning models to load (default: ['deepsurv', 'deephit']).

    Returns:
        dict: A dictionary containing loaded models.
        pd.DataFrame: The result table DataFrame.
    """
    # Define paths
    models_folder = os.path.join(path_dir,  "models")
    results_file = os.path.join(path_dir, "results", "result.csv")
    models = {}

    # Load results table
    if os.path.exists(results_file):
        table_final = pd.read_csv(results_file)
    else:
        raise FileNotFoundError(f"Results file not found at {results_file}.")

    # Load machine learning models
    files_ml = [p for p in os.listdir(models_folder) if '_ML.pkl' in p]
    for n in files_ml:
        name = n.replace('_ML', '').replace('.pkl', '')
        with open(os.path.join(models_folder, n), 'rb') as f:
            models[name] = pickle.load(f)

    # Load deep learning models using parameters from table_final
    files_dl = [p for p in os.listdir(models_folder) if '_DL.pt' in p]
    for model_name in files_dl:
        model_file = f"{model_name}"
        try:
          if model_name == 'deepsurv_DL.pt':
              # Extract parameters for DeepSurv
              params = table_final[table_final['model'] == "deepsurv"].dropna(axis=1) \
                  .drop(['model','lr','batch_size'] + [c for c in table_final.columns if 'score' in c], axis=1) \
                  .iloc[0].to_dict()
              #print(params)
              models['deepsurv'] = load_model(model_file, models_folder, CoxPH, len(cols_x), 1, params)
          elif model_name == 'deephit_DL.pt':
              # Extract parameters for DeepHit
              params = table_final[table_final['model'] == "deephit"].dropna(axis=1) \
                  .drop(['model', 'lr', 'batch_size', 'discrete'] + [c for c in table_final.columns if 'score' in c], axis=1) \
                  .iloc[0].to_dict()
              models['deephit'] = load_model(model_file, models_folder, DeepHitSingle, len(cols_x), 1, params)
          else:
              continue

        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    table_final = table_final[['model',  'score_train' , 'score_test']]
    return models, table_final

def get_bier_score(df, y_train, y_test, survs, times, col_target="time2event", with_benchmark=True):
    """
    Compute Brier scores for survival models with optional benchmarks.

    Args:
        df (pd.DataFrame): DataFrame containing survival data.
        y_train (structured array): Training survival data.
        y_test (structured array): Testing survival data.
        survs (list): List of survival functions for test samples.
        times (np.ndarray): Time points for evaluation.
        col_target (str): Column name for the time-to-event variable.
        with_benchmark (bool): Whether to include benchmarks (Kaplan-Meier, random).

    Returns:
        dict: Dictionary of integrated Brier scores for each prediction type.
    """
    # Ensure survs contains callable functions
    if not all(callable(fn) for fn in survs):
        raise TypeError("All elements in 'survs' must be callable survival functions.")

    if with_benchmark:
        # Compute Kaplan-Meier function for benchmarks
        km_func = StepFunction(
            *kaplan_meier_estimator(df["censored"].astype(bool), df[col_target])
        )
        
        # Create prediction sets
        preds = {
            'estimator': np.row_stack([fn(times) for fn in survs]),
            'random': 0.5 * np.ones((df.shape[0], times.shape[0])),
            'kaplan_meier': np.tile(km_func(times), (df.shape[0], 1))
        }
    else:
        preds = {'estimator': np.row_stack([fn(times) for fn in survs])}

    # Compute integrated Brier scores
    scores = {}
    for k, v in preds.items():
        scores[k] = integrated_brier_score(y_train, y_test, v, times)

    return scores

def convert_risk_to_survival(risks, times):
    """
    Convert risk scores from FastSurvivalSVM to StepFunction survival probabilities.

    Args:
        risks (np.ndarray): Risk scores from FastSurvivalSVM.
        times (np.ndarray): Time points for evaluation.

    Returns:
        list: List of StepFunction objects representing survival probabilities.
    """
    # Normalize risks to [0, 1] range
    max_risk = np.max(risks)
    normalized_risks = risks / max_risk  # Higher risks imply lower survival probabilities

    # Generate survival probabilities over time
    survival_curves = np.exp(-np.outer(normalized_risks, times / np.max(times)))

    # Create StepFunction for each survival curve
    step_functions = [
        StepFunction(x=times, y=curve) for curve in survival_curves
    ]

    return step_functions

def get_integrated_brier_score(models, X_train, X_test, y_train, y_test, cols_x, times, col_target="time2event"):
    """
    Compute the integrated Brier score for multiple survival models, including DeepSurv and DeepHit.

    Args:
        models (dict): Dictionary of survival models with model names as keys.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.DataFrame): Training target survival data.
        y_test (pd.DataFrame): Testing target survival data.
        cols_x (list): List of feature column names.
        times (np.ndarray): Time points for evaluation.
        col_target (str): Column name for the target survival time.

    Returns:
        dict: A dictionary of integrated Brier scores for each model.
    """
    survs = {}
    integrated_scores = {}
    print("Processing Kaplan-Meier and Random benchmarks... ???? ")

    # Get the maximum observed time in the training set
    max_time_train = np.max(y_train[col_target])

    # Filter the test set to ensure times do not exceed the maximum time in the training set
    valid_test_indices = y_test[col_target] <= max_time_train
    y_test_filtered = y_test[valid_test_indices]
    X_test_filtered = X_test[valid_test_indices]

    # Filter times to be within range
    valid_times = times[times <= max_time_train]

    # Compute Kaplan-Meier benchmark
    km_func = StepFunction(
        *kaplan_meier_estimator(y_train["censored"].astype(bool), y_train[col_target])
    )
    kaplan_meier_preds = np.tile(km_func(valid_times), (len(X_test_filtered), 1))  # KM survival probabilities for all test samples

    # Random survival probabilities benchmark
    random_preds = 0.5 * np.ones((len(X_test_filtered), len(valid_times)))

    # Add Kaplan-Meier and Random to integrated_scores
    integrated_scores["kaplan_meier"] = integrated_brier_score(y_train, y_test_filtered, kaplan_meier_preds, valid_times)
    integrated_scores["random"] = integrated_brier_score(y_train, y_test_filtered, random_preds, valid_times)

    for name in models:
        print(f"Processing model: {name}")

        if name == 'deepsurv':
            # DeepSurv specific computation
            model = models[name]
            get_target = lambda df: (df[col_target].values, df['censored'].values)

            # Compute baseline hazards
            _ = model.compute_baseline_hazards(
                np.array(X_train[cols_x]).astype(np.float32), get_target(X_train)
            )

            # Predict survival probabilities as a DataFrame
            survs[name] = model.predict_surv_df(np.array(X_test[cols_x]).astype(np.float32))

            # Evaluate Brier scores
            ev = EvalSurv(survs[name], y_test[col_target], y_test['censored'], censor_surv='km')
            integrated_scores[name] = ev.integrated_brier_score(times)
            #print(f"Integrated Brier Score for {name}: {integrated_scores[name]:.4f}")

        elif name == 'deephit':
            # DeepHit specific computation
            model = models[name]

            # Predict survival probabilities as a DataFrame
            survs[name] = model.predict_surv_df(np.array(X_test[cols_x]).astype(np.float32))

            # Evaluate Brier scores
            ev = EvalSurv(survs[name], y_test[col_target], y_test['censored'], censor_surv='km')
            integrated_scores[name] = ev.integrated_brier_score(times)
            #print(f"Integrated Brier Score for {name}: {integrated_scores[name]:.4f}")

        elif hasattr(models[name], 'predict_survival_function'):
            # For models with survival function predictions
            #survs[name] = models[name].predict_survival_function(X_test[cols_x])
            #brier_scores = get_bier_score(X_test, y_train, y_test, survs[name], times, col_target, with_benchmark=True)
            survs[name] = models[name].predict_survival_function(X_test_filtered[cols_x])
            brier_scores = get_bier_score(X_test_filtered, y_train, y_test_filtered, survs[name], times, col_target, with_benchmark=True)
            integrated_scores[name] = brier_scores['estimator']

        elif isinstance(models[name], FastSurvivalSVM):
            # For FastSurvivalSVM or similar models
            risks = models[name].predict(X_test_filtered[cols_x])
            survs[name] = convert_risk_to_survival(risks, times)
            brier_scores = get_bier_score(X_test_filtered, y_train, y_test_filtered, survs[name], times, col_target, with_benchmark=True)
            integrated_scores[name] = brier_scores['estimator']

        else:
            raise AttributeError(f"Model '{name}' does not support survival predictions.")

    integrated_scores = {name: round(score, 5) for name, score in integrated_scores.items()}
    return integrated_scores
