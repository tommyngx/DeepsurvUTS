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
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import requests
from matplotlib.font_manager import FontProperties

# Add MLPVanilla and set to the safe globals for torch.load
torch.serialization.add_safe_globals([tt.practical.MLPVanilla, set])

# Download and set the custom font
font_url = 'https://github.com/tommyngx/style/blob/main/Poppins.ttf?raw=true'
font_path = 'Poppins.ttf'
response = requests.get(font_url)
with open(font_path, 'wb') as f:
    f.write(response.content)
# Set the custom font with size
font_prop = FontProperties(fname=font_path, size=19)

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
    
    # Load the model with weights_only=True and map to CPU
    try:
        model.load_net(os.path.join(path, filename), weights_only=False, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading {filename} with weights_only=True: {e}")
        print("Attempting to load with weights_only=False...")
        model.load_net(os.path.join(path, filename), weights_only=False, map_location=torch.device('cpu'))

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
    #print("Processing Kaplan-Meier and Random benchmarks... ???? ")

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
        #print(f"Processing model: {name}")

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

def get_brier_curves(models, X_train, X_test, y_train, y_test, cols_x, times=np.arange(1, 20)):
    """
    Compute Brier score curves for 'gboost', 'cox_ph', 'deepsurv', 'deephit', 'svm', and 'rsf' models over a range of times.

    Args:
        models (dict): Dictionary containing supported models.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.DataFrame): Training target survival data.
        y_test (pd.DataFrame): Testing target survival data.
        cols_x (list): List of feature column names.
        times (np.ndarray): Time points for evaluation.

    Returns:
        pd.DataFrame: DataFrame of Brier score curves for all models.
    """
    # Get the maximum observed time in the training set
    max_time_train = np.max(y_train["time2event"])

    # Filter the test set to ensure times do not exceed the maximum time in the training set
    valid_test_indices = y_test["time2event"] <= max_time_train
    y_test_filtered = y_test[valid_test_indices]
    X_test_filtered = X_test[valid_test_indices]

    brier_curves = None

    for i, name in enumerate(['gboost', 'cox_ph', 'deepsurv', 'deephit', 'svm', 'rsf']):
        if name not in models:
            print(f"Skipping {name}: Model not found.")
            continue

        print(f"Processing model: {name}")
        model = models[name]

        if name == 'deepsurv':
            # DeepSurv specific computation
            survs = model.predict_surv_df(np.array(X_test_filtered[cols_x]).astype(np.float32))
            ev = EvalSurv(survs, y_test_filtered["time2event"], y_test_filtered['censored'], censor_surv='km')
            scores = ev.brier_score(times)

        elif name == 'deephit':
            # DeepHit specific computation
            survs = model.predict_surv_df(np.array(X_test_filtered[cols_x]).astype(np.float32))
            ev = EvalSurv(survs, y_test_filtered["time2event"], y_test_filtered['censored'], censor_surv='km')
            scores = ev.brier_score(times)

        elif name == 'svm':
            # SVM (FastSurvivalSVM) specific computation
            risks = model.predict(X_test_filtered[cols_x])
            survs = convert_risk_to_survival(risks, times)
            scores = []
            for t in times:
                preds = [fn(t) for fn in survs]
                _, score = brier_score(y_train, y_test_filtered, preds, t)
                scores.append(score[0])

        elif name == 'rsf':
            # Random Survival Forest specific computation
            survs = model.predict_survival_function(X_test_filtered[cols_x])
            scores = []
            for t in times:
                preds = [fn(t) for fn in survs]
                _, score = brier_score(y_train, y_test_filtered, preds, t)
                scores.append(score[0])

        else:
            # For 'gboost' and 'cox_ph' models
            survs = model.predict_survival_function(X_test_filtered[cols_x])
            scores = []
            for t in times:
                preds = [fn(t) for fn in survs]
                _, score = brier_score(y_train, y_test_filtered, preds, t)
                scores.append(score[0])

        # Create a DataFrame for this model
        scores_df = pd.DataFrame({'time': times, name: scores})

        # Merge with the main DataFrame
        brier_curves = scores_df if brier_curves is None else brier_curves.merge(scores_df, on='time')

    return brier_curves

def plot_brier_curves_with_color_list(brier_curves, model_name_map=None, save_folder=None):
    """
    Plot Brier score curves using Matplotlib with y-axis in percentage format, markers for each data point,
    and a predefined list of colors. Optionally replace model names using a mapping.

    Args:
        brier_curves (pd.DataFrame): DataFrame containing Brier scores over time.
                                     The 'time' column contains the x-axis values.
        model_name_map (dict, optional): A dictionary to map original model names to display names.
                                         For example, {'deepsurv': 'DeepSurv', 'cox_ph': 'Cox Proportional Hazard'}.
        save_folder (str, optional): Folder to save the plot as a .png file.
    """
    plt.figure(figsize=(8, 6))

    # Define the color list
    color_list = [
        "#2ca02c", "#8c564b", "#9467bd", "#d62728", "#ff7f0e",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Ensure the number of models does not exceed the color list length
    models = [m for m in brier_curves.columns if m != 'time']

    # Plot each column (except 'time') against the time column with assigned colors
    for idx, m in enumerate(models):
        color = color_list[idx]  # Assign color from the list

        # Use model_name_map if provided, otherwise use original names
        display_name = model_name_map.get(m, m) if model_name_map else m

        # Plot the curve and scatter points
        plt.plot(brier_curves['time'], brier_curves[m] * 100, label=display_name, linestyle='-', color=color)
        plt.scatter(brier_curves['time'], brier_curves[m] * 100, marker='o', s=20, color=color)

    # Customize the plot
    plt.title("Brier Score Curves", fontproperties=font_prop, pad=20)
    plt.xlabel("Time (years)", fontsize=14, fontproperties=font_prop)
    plt.ylabel("Brier Score (%)", fontsize=14, fontproperties=font_prop)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(decimals=0))  # Format y-axis as percentages without decimals

    # Customize legend with a white background
    legend = plt.legend()
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot if save_folder is provided
    if save_folder:
        save_path = f"{save_folder}/brier_curves.png"
        plt.savefig(save_path, format='png')

    plt.show()