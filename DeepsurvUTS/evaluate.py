import itertools
import pandas as pd
import numpy as np
import datetime
import time
import os
import torch

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

from sksurv.metrics import concordance_index_censored
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.metrics import brier_score, integrated_brier_score
from pycox.evaluation import EvalSurv
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from lifelines import KaplanMeierFitter
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
import shap
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.ensemble import RandomSurvivalForest

from sksurv.functions import StepFunction
from sksurv.nonparametric import kaplan_meier_estimator
from matplotlib.ticker import FuncFormatter


def compute_score(censored, target, prediction, sign):
    return concordance_index_censored(list(censored.astype(bool)), target, sign*prediction)[0]

def compute_score_model(preds_df, col_var, col_pred, col_target, sign, metric="cindex", times = None):
    
    scores = {}
    
    for k in preds_df[col_var].unique():
    
        tmp = preds_df[preds_df[col_var]==k]
        scores[k] = compute_score(tmp.censored, tmp[col_target], tmp[col_pred], sign)
        
    scores = pd.DataFrame.from_dict(scores, orient='index').reset_index().rename(
        columns={'index':col_var, 0:col_pred})
        
    return scores

def get_distrib(data, col_var, name):
    
    cols_x = [c for c in data.columns if c !=col_var]
    
    distrib = data.groupby(col_var,as_index=False)[cols_x[0]].count()
    
    distrib[f'perc_{name}'] = distrib[cols_x[0]]/data.shape[0]*100
    distrib.drop(cols_x[0], axis=1, inplace=True)
    
    return distrib
    
def plot_score(scores_df, col_var, models_name):

    scores_graph = pd.melt(
        scores_df[[col_var]+models_name], 
        id_vars=[col_var], value_name='score', var_name='model')

    scores_graph[col_var] = scores_graph[col_var].astype(str)
    scores_graph['score_round'] = scores_graph.score.round(3).astype('str')
    
    fig = px.bar(
        scores_graph, x='model', y='score', 
        color = col_var, barmode='group',
        text = 'score_round',
        color_discrete_sequence = ['royalblue','lightgrey']
    )

    fig.update_traces(textposition='outside')

    fig.update_layout(
        dict(
            title = "{} - {}% of positive classes".format(
                col_var.capitalize(), 
                round(scores_df[scores_df[col_var]==1]['perc_train'].iloc[0])
            ),
            xaxis={'title' : 'Model'}, 
            yaxis={'title' : 'Concordance index', 'range': [0,1]},
        )
    )
    
    return fig

def plot_concordance_bar_chart(table_final, model_name_map, y_labels=('train', 'test'), width=0.35, save_folder=None):
    """
    Plot a bar chart of concordance index scores for train and test datasets.

    Args:
        table_final (pd.DataFrame): DataFrame containing model names and scores.
        model_name_map (dict): Mapping of internal model names to display names.
        y_labels (tuple): Labels for the y values (default: ('train', 'test')).
        width (float): Width of the bars (default: 0.35).
        save_folder (str, optional): Folder to save the plot as a .png file.
    """
    # Map model names
    table_final['model_mapped'] = table_final['model'].map(model_name_map)

    # Data preparation
    x_labels = table_final['model_mapped']
    x = range(len(x_labels))  # Numeric positions for models

    # Extract scores
    scores = {label: table_final[f'score_{label}'].apply(lambda x: round(x, 3)) for label in y_labels}

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add bars for each dataset (train/test)
    bars = {}
    for i, label in enumerate(y_labels):
        bars[label] = ax.bar([p + i * width for p in x], scores[label], width, label=label)

    # Add text labels above bars
    for label, bar_group in bars.items():
        for bar in bar_group:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.3f}", ha='center', va='bottom')

    # Customize the plot
    ax.set_title('Concordance Index by Model', fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Concordance Index', fontsize=12)
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylim(0, 1)  # Set y-axis range
    ax.legend()

    # Show grid and plot
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot if save_folder is provided
    if save_folder:
        save_path = f"{save_folder}/results/concordance.png"
        plt.savefig(save_path, format='png')

    plt.show()


from sksurv.metrics import integrated_brier_score
from sksurv.functions import StepFunction
from sksurv.nonparametric import kaplan_meier_estimator
import numpy as np


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

# Define convert_risk_to_survival
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

    # Compute Kaplan-Meier benchmark
    km_func = StepFunction(
        *kaplan_meier_estimator(y_train["censored"].astype(bool), y_train[col_target])
    )
    kaplan_meier_preds = np.tile(km_func(times), (len(X_test), 1))  # KM survival probabilities for all test samples

    # Random survival probabilities benchmark
    random_preds = 0.5 * np.ones((len(X_test), len(times)))

    # Add Kaplan-Meier and Random to integrated_scores
    integrated_scores["kaplan_meier"] = integrated_brier_score(y_train, y_test, kaplan_meier_preds, times)
    integrated_scores["random"] = integrated_brier_score(y_train, y_test, random_preds, times)

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
            survs[name] = models[name].predict_survival_function(X_test[cols_x])
            brier_scores = get_bier_score(X_test, y_train, y_test, survs[name], times, col_target, with_benchmark=True)
            integrated_scores[name] = brier_scores['estimator']

        elif isinstance(models[name], FastSurvivalSVM):
            # For FastSurvivalSVM or similar models
            risks = models[name].predict(X_test[cols_x])
            survs[name] = convert_risk_to_survival(risks, times)
            brier_scores = get_bier_score(X_test, y_train, y_test, survs[name], times, col_target, with_benchmark=True)
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
    brier_curves = None

    for i, name in enumerate(['gboost', 'cox_ph', 'deepsurv', 'deephit', 'svm', 'rsf']):
        if name not in models:
            print(f"Skipping {name}: Model not found.")
            continue

        print(f"Processing model: {name}")
        model = models[name]

        if name == 'deepsurv':
            # DeepSurv specific computation
            survs = model.predict_surv_df(np.array(X_test[cols_x]).astype(np.float32))
            ev = EvalSurv(survs, y_test["time2event"], y_test['censored'], censor_surv='km')
            scores = ev.brier_score(times)

        elif name == 'deephit':
            # DeepHit specific computation
            survs = model.predict_surv_df(np.array(X_test[cols_x]).astype(np.float32))
            ev = EvalSurv(survs, y_test["time2event"], y_test['censored'], censor_surv='km')
            scores = ev.brier_score(times)

        elif name == 'svm':
            # SVM (FastSurvivalSVM) specific computation
            risks = model.predict(X_test[cols_x])
            survs = convert_risk_to_survival(risks, times)
            scores = []
            for t in times:
                preds = [fn(t) for fn in survs]
                _, score = brier_score(y_train, y_test, preds, t)
                scores.append(score[0])

        elif name == 'rsf':
            # Random Survival Forest specific computation
            survs = model.predict_survival_function(X_test[cols_x])
            scores = []
            for t in times:
                preds = [fn(t) for fn in survs]
                _, score = brier_score(y_train, y_test, preds, t)
                scores.append(score[0])

        else:
            # For 'gboost' and 'cox_ph' models
            survs = model.predict_survival_function(X_test[cols_x])
            scores = []
            for t in times:
                preds = [fn(t) for fn in survs]
                _, score = brier_score(y_train, y_test, preds, t)
                scores.append(score[0])

        # Create a DataFrame for this model
        scores_df = pd.DataFrame({'time': times, name: scores})

        # Merge with the main DataFrame
        brier_curves = scores_df if brier_curves is None else brier_curves.merge(scores_df, on='time')

    return brier_curves


def plot_brier_scores(df, highlight_mean=False, model_name_map=None, save_folder=None):
    """
    Bar plot of Brier scores for models from a DataFrame with additional options.

    Args:
        df (pd.DataFrame): DataFrame with models as the index and Brier scores as a column.
        highlight_mean (bool): If True, highlights specific models (e.g., 'deepsurv', 'deephit') in royal blue, others in dark grey.
        save_folder (str, optional): Folder to save the plot as a .png file.
    """
    # Check if df is not a DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame.from_dict(df, orient='index')\
                        .rename(columns={0: 'integrated_brier_score'})

    # Map model names if model_name_map is provided
    if model_name_map:
        df.index = df.index.map(lambda x: model_name_map.get(x, x))  # Replace names using the map, fallback to original

    # Sort the DataFrame by Brier scores
    df_sorted = df.sort_values(by=df.columns[0], ascending=False)  # Sorting by the first column

    # Colors based on highlighting option
    if highlight_mean:
        # Highlight 'deepsurv' and 'deephit' (case and spacing insensitive)
        colors = [
            "royalblue" if "deepsurv" in model.lower().replace(" ", "") or "deephit" in model.lower().replace(" ", "")
            else "darkgrey" for model in df_sorted.index
        ]
    else:
        colors = ['skyblue', 'orange', 'green', 'red', 'purple', 'pink'][:len(df_sorted)]

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_sorted.index, df_sorted.iloc[:, 0], color=colors)

    # Add score labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, height + 0.01,  # Position slightly above the bar
            f"{height:.3f}", ha='center', va='bottom', fontsize=10
        )

    # Customize plot
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Brier Score', fontsize=12)
    plt.title('Brier Scores for Each Model', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Optional: Add a horizontal grid for better readability
    plt.tight_layout()

    # Save the plot if save_folder is provided
    if save_folder:
        save_path = f"{save_folder}/results/brier_scores.png"
        plt.savefig(save_path, format='png')

    plt.show()


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
        "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Ensure the number of models does not exceed the color list length
    models = [m for m in brier_curves.columns if m != 'time']
    if len(models) > len(color_list):
        raise ValueError(f"Number of models ({len(models)}) exceeds available colors ({len(color_list)})")

    # Plot each column (except 'time') against the time column with assigned colors
    for idx, m in enumerate(models):
        color = color_list[idx]  # Assign color from the list

        # Use model_name_map if provided, otherwise use original names
        display_name = model_name_map.get(m, m) if model_name_map else m

        # Plot the curve and scatter points
        plt.plot(brier_curves['time'], brier_curves[m] * 100, label=display_name, linestyle='-', color=color)
        plt.scatter(brier_curves['time'], brier_curves[m] * 100, marker='o', s=20, color=color)

    # Customize the plot
    plt.title("Brier Score Curves", fontsize=14)
    plt.xlabel("Time (years)", fontsize=12)
    plt.ylabel("Brier Score (%)", fontsize=12)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(decimals=0))  # Format y-axis as percentages without decimals

    # Customize legend with a white background
    legend = plt.legend()
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot if save_folder is provided
    if save_folder:
        save_path = f"{save_folder}/results/brier_curves.png"
        plt.savefig(save_path, format='png')

    plt.show()



def plot_kaplan_meier_with_models(y_time, y_censored, models, X_test, models_to_plot, cols_x, time_points, model_name_map=None, title='Kaplan-Meier vs Models Survival Curve', save_folder=None):
    """
    Plot Kaplan-Meier survival curve alongside survival curves from selected models with specified colors.

    Args:
        y_time (pd.Series): Time-to-event data.
        y_censored (pd.Series): Censoring indicator (1 = event, 0 = censored).
        models (dict): Dictionary of trained survival models.
        X_test (pd.DataFrame): Feature data for the test set.
        models_to_plot (list): List of model names to include in the plot.
        cols_x (list): List of feature column names used during model training.
        time_points (np.array): Time points for evaluating survival probabilities.
        model_name_map (dict, optional): Mapping of model names to display-friendly names.
        title (str): Title for the plot.
        save_folder (str, optional): Folder to save the plot as a .png file.
    """
    # Kaplan-Meier estimator
    kmf = KaplanMeierFitter()
    kmf.fit(durations=y_time, event_observed=y_censored)

    # Generate survival probabilities for selected models
    survs = {}
    for model_name in models_to_plot:
        if model_name not in models:
            print(f"Model '{model_name}' not found. Skipping.")
            continue
        model = models[model_name]
        if model_name in ['deepsurv', 'deephit']:
            survs[model_name] = model.predict_surv_df(np.array(X_test[cols_x]).astype(np.float32))
        elif hasattr(model, 'predict_survival_function'):
            survs[model_name] = model.predict_survival_function(X_test[cols_x])

        # Convert SVM risk scores into StepFunction survival probabilities
        elif model_name == 'svm':
            risks = model.predict(X_test[cols_x])  # Predict risk scores
            risks = np.asarray(risks)  # Ensure risks are in array format

            # Normalize risks to [0, 1]
            survival_probs = 1 - np.clip((risks - risks.min()) / (risks.max() - risks.min()), 0, 1)

            # Convert normalized survival probabilities to StepFunctions
            survs[model_name] = [
                StepFunction(time_points, np.full(len(time_points), prob)) for prob in survival_probs
            ]
        else:
            print(f"Model '{model_name}' does not support survival probabilities. Skipping.")

    # Plot Kaplan-Meier survival curve
    plt.figure(figsize=(8, 5))
    kmf.plot_survival_function(label='Kaplan-Meier', ci_show=False, color='black', linestyle='--')

    # Define a list of colors for models
    color_list = [
        "#f0614a", "#11cd9a", "#ab63fa", "#8c564b", '#1f77b4', '#e377c2',
        "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Create a color map for models
    color_map = {model_name: color_list[idx % len(color_list)] for idx, model_name in enumerate(models_to_plot)}

    # Adjust time_points based on maximum time range in survival curves
    max_time = 25
    for model_name, model_surv in survs.items():
        if isinstance(model_surv, np.ndarray) and isinstance(model_surv[0], StepFunction):
            max_time = min(max_time, min([fn.domain[1] for fn in model_surv]))
        elif isinstance(model_surv, pd.DataFrame):
            max_time = min(max_time, model_surv.index.max())
    time_points = np.linspace(0, max_time, len(time_points))

    # Plot survival curves for models
    for model_name, model_surv in survs.items():
        display_name = model_name_map.get(model_name, model_name) if model_name_map else model_name
        color = color_map[model_name]
        
        if isinstance(model_surv, np.ndarray) and isinstance(model_surv[0], StepFunction):
            # StepFunction survival curves
            evaluated_survs = np.array([fn(time_points) for fn in model_surv])
            mean_surv = evaluated_survs.mean(axis=0)
            plt.plot(
                time_points, mean_surv,
                label=f"{display_name} (Mean Curve)",
                linestyle='-', alpha=0.8, color=color
            )
        elif isinstance(model_surv, pd.DataFrame):
            # DataFrame survival curves
            mean_surv = model_surv.mean(axis=1)
            plt.plot(
                model_surv.index, mean_surv,
                label=f"{display_name} (Mean Curve)",
                linestyle='-', alpha=0.8, color=color
            )
        else:
            print(f"Unsupported format for model '{model_name}'.")

    # Customize the plot
    plt.xlim(0, max_time)
    plt.ylim(0, 1)
    plt.title(title, fontsize=14)
    plt.xlabel('Time (years)', fontsize=12)
    plt.ylabel('Survival Probability', fontsize=12)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))  # Percentage format for survival probability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Customize legend with a white background
    legend = plt.legend(loc='best', fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    plt.tight_layout()

    # Save the plot if save_folder is provided
    if save_folder:
        save_path = f"{save_folder}/results/kaplan_meier.png"
        plt.savefig(save_path, format='png')

    plt.show()


def plot_10_year_calibration_curve(models_to_plot, all_probs_df, time_col, censored_col, threshold=10, title="10-Year Calibration Curve", save_folder=None):
    """
    Plot calibration curves for models, ensuring consistent and repeatable results.

    Args:
        models_to_plot (list): List of model names to include in the calibration plot.
        all_probs_df (pd.DataFrame): DataFrame containing predicted probabilities and survival data.
        time_col (str): Name of the column representing time-to-event.
        censored_col (str): Name of the column representing censoring status (1 = event, 0 = censored).
        threshold (float): Time threshold (e.g., 10 years).
        title (str): Title for the plot.
        save_folder (str, optional): Folder to save the plot as a .png file.
    """
    np.random.seed(42)
    # Step 1: Create Actual Outcome Column
    all_probs_df['Actual Outcome'] = ((all_probs_df[time_col] <= threshold) & (all_probs_df[censored_col] == 1)).astype(int)

    # Reverse the values in the 'Actual Outcome' column
    #all_probs_df['Actual Outcome'] = 1 - all_probs_df['Actual Outcome']

    # Define color list
    color_list = [
        "#f0614a", "#11cd9a", "#ab63fa", "#8c564b", '#1f77b4', '#e377c2',
        "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Ensure models and colors align
    color_map = {model_name: color_list[idx] for idx, model_name in enumerate(models_to_plot)}

    plt.figure(figsize=(8, 6))

    # Step 2: Loop through models and plot calibration curves
    for model_name in models_to_plot:
        if model_name in all_probs_df.columns:
            # Get predicted probabilities and actual outcomes
            predicted = all_probs_df[model_name].values
            actual = all_probs_df['Actual Outcome'].values

            # Ensure deterministic binning for calibration curve
            sorted_indices = np.argsort(predicted)
            predicted_sorted = predicted[sorted_indices]
            actual_sorted = actual[sorted_indices]

            # Compute calibration curve
            prob_true, prob_pred = calibration_curve(
                y_true=actual_sorted,
                y_prob=predicted_sorted,
                n_bins=10,
                strategy='uniform'
            )

            prob_true = 1 - prob_true
            # Plot calibration curve
            plt.plot(
                prob_pred, prob_true,
                marker='o', label=model_name,
                color=color_map[model_name]
            )

    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)

    # Customize plot
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted Probability (10 years)", fontsize=12)
    plt.ylabel("Observed Proportion (10 years)", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    # Customize legend with a white background
    legend = plt.legend()
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot if save_folder is provided
    if save_folder:
        save_path = f"{save_folder}/results/calibration_curve.png"
        plt.savefig(save_path, format='png')

    plt.show()



def plot_roc_curve(models_to_plot, all_probs_df, time_col, censored_col, threshold=10, title="10-Year ROC Curve", model_name_map=None, save_folder=None):
    """
    Plot ROC curves for models with support for mapped names, ensuring colors follow a specific order.

    Args:
        models_to_plot (list): List of model names to include in the ROC plot.
        all_probs_df (pd.DataFrame): DataFrame containing predicted probabilities and survival data.
        time_col (str): Name of the column representing time-to-event.
        censored_col (str): Name of the column representing censoring status (1 = event, 0 = censored).
        threshold (float): Time threshold (e.g., 10 years).
        title (str): Title for the plot.
        model_name_map (dict): Dictionary mapping internal model names to display-friendly names.
        save_folder (str, optional): Folder to save the plot as a .png file.
    """
    # Step 1: Create Actual Outcome Column
    all_probs_df['Actual Outcome'] = ((all_probs_df[time_col] <= threshold) & (all_probs_df[censored_col] == 1)).astype(int)

    # Define color list
    color_list = [
        "#f0614a", "#11cd9a", "#ab63fa", "#8c564b", '#1f77b4', '#e377c2',
        "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Ensure models and colors align
    color_map = {model_name: color_list[idx % len(color_list)] for idx, model_name in enumerate(models_to_plot)}

    plt.figure(figsize=(7, 5))

    # Step 2: Loop through models and plot ROC curves
    for model_name in models_to_plot:
        if model_name in all_probs_df.columns:
            # Map model name if mapping is provided
            display_name = model_name_map.get(model_name, model_name) if model_name_map else model_name

            # Get predicted probabilities and actual outcomes
            predicted = all_probs_df[model_name]
            actual = all_probs_df['Actual Outcome']

            actual = 1 - actual
            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(y_true=actual, y_score=predicted)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.plot(
                fpr, tpr,
                label=f"{display_name} (AUC = {roc_auc:.2f})",
                color=color_map[model_name]
            )

    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', alpha=0.7)

    # Customize plot
    plt.title(title, fontsize=14)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    # Customize legend with a white background
    legend = plt.legend()
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot if save_folder is provided
    if save_folder:
        save_path = f"{save_folder}/results/roc_curve.png"
        plt.savefig(save_path, format='png')

    plt.show()


def plot_dca_results(df, models_to_plot=None, title="Decision Curve Analysis (DCA)", xlabel="Threshold Probability (%)",
                     ylabel="Net Benefit", x_limits=None, y_limits=None, model_name_map=None, save_folder=None):
    """
    Plot Decision Curve Analysis (DCA) results from a DataFrame with options for x-axis limits, y-axis limits,
    and model name mapping.

    Args:
        df (pd.DataFrame): DataFrame with DCA results, containing columns like:
                           'model', 'threshold', 'net_benefit', etc.
        models_to_plot (list, optional): Specific models to include in the plot. If None, plot all models.
        title (str): Title for the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        x_limits (tuple, optional): Tuple specifying x-axis limits as (x_min, x_max).
        y_limits (tuple, optional): Tuple specifying y-axis limits as (y_min, y_max).
        model_name_map (dict, optional): Dictionary for mapping internal model names to display-friendly names.
        save_folder (str, optional): Folder to save the plot as a .png file.
    """
    plt.figure(figsize=(8, 5))  # Updated figsize

    # Filter the DataFrame for selected models
    if models_to_plot:
        df = df[df['model'].isin(models_to_plot)]

    # Map model names if model_name_map is provided
    if model_name_map:
        df['model'] = df['model'].replace(model_name_map)

    # Ensure models are ordered for consistent colors
    models = df['model'].unique()

    # Define custom color list
    color_list = [
        "#f0614a", "#ab63fa", "#11cd9a", "#8c564b", '#1f77b4', '#e377c2',
        "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    color_map = {model: color_list[idx % len(color_list)] for idx, model in enumerate(models)}

    # Plot net benefit for each model
    for model in models:
        model_data = df[df['model'] == model]
        plt.plot(
            model_data['threshold'] * 100,  # Convert threshold to percentage
            model_data['net_benefit'],
            label=model,
            color=color_map[model],
            linestyle='-'
        )

    # Add "Treat All" and "Treat None" references if available
    if 'All' in df['model'].values:
        all_data = df[df['model'] == 'All']
        plt.plot(
            all_data['threshold'] * 100,
            all_data['net_benefit'],
            label='Treat All',
            color='black',
            linestyle='--'
        )
    if 'None' in df['model'].values:
        none_data = df[df['model'] == 'None']
        plt.plot(
            none_data['threshold'] * 100,
            none_data['net_benefit'],
            label='Treat None',
            color='gray',
            linestyle='--'
        )

    # Customize plot
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.axhline(0, color="black", linestyle="--", alpha=0.6)  # Reference line at 0

    # Set x-axis and y-axis limits if provided
    if x_limits:
        plt.xlim(x_limits)
    if y_limits:
        plt.ylim(y_limits)

    # Add legend with white background and black edge
    legend = plt.legend(loc="best", fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    # Add grid and adjust layout
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot if save_folder is provided
    if save_folder:
        save_path = f"{save_folder}/results/dca_results.png"
        plt.savefig(save_path, format='png')

    plt.show()

def reverse_scaling(X_scaled, scaler, feature_names):
    X_original = scaler.inverse_transform(X_scaled)
    return pd.DataFrame(X_original, columns=feature_names)

def plot_shap_values_for_ml_model(model, X_train, y_train, X_val, scaler, cols_x, save_folder=None):
    """
    Compute and plot SHAP values for a machine learning model.

    Args:
        model: Trained machine learning model.
        X_train (pd.DataFrame): Scaled training feature data.
        y_train (pd.Series): Training target values.
        X_val (pd.DataFrame): Scaled validation feature data.
        scaler: Fitted scaler used for preprocessing features.
        cols_x (list): List of feature column names.
        save_folder (str, optional): Folder to save the SHAP plots as .png files.

    Returns:
        None
    """
    # Reverse scaling for SHAP interpretation
    X_train_original = reverse_scaling(X_train[cols_x], scaler, feature_names=cols_x)
    X_val_original = reverse_scaling(X_val[cols_x], scaler, feature_names=cols_x)

    # Fit the model
    print(f"Training model: {model.__class__.__name__}...")
    model.fit(X_train_original, y_train)

    # Initialize SHAP Explainer
    print("Initializing SHAP explainer...")
    explainer = shap.Explainer(model.predict, X_train_original)

    # Compute SHAP values for validation dataset
    print("Computing SHAP values for the validation dataset...")
    shap_values_val = explainer(X_val_original)

    # Plot SHAP waterfall plot for the first validation sample
    print("Generating SHAP waterfall plot for the first validation sample...")
    #plt.figure()
    shap.plots.waterfall(shap_values_val[0])
    #shap.waterfall_plot(shap_values_val[0])
    if save_folder:
        shap.plots.waterfall(shap_values_val[0], show=False)
        save_path = f"{save_folder}/results/shap_waterfall.png"
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=200)
        print(f"SHAP waterfall plot saved at: {save_path}")

    # Plot SHAP summary plot for validation dataset
    print("Generating SHAP summary plot for validation dataset...")
    #shap.summary_plot(shap_values_val, X_val_original)
    #if save_folder:
    #    save_path = f"{save_folder}/results/shap_summary.png"
    #    plt.savefig(save_path, format='png')
    #    print(f"SHAP summary plot saved at: {save_path}")

    # Plot SHAP dependence plot for the most important feature
    top_feature = X_val_original.columns[np.argmax(shap_values_val.values.mean(axis=0))]
    print(f"Generating SHAP dependence plot for the top feature: {top_feature}")
    #shap.dependence_plot(top_feature, shap_values_val.values, X_val_original)
    #if save_folder:
    #    save_path = f"{save_folder}/results/shap_dependence.png"
    #    plt.savefig(save_path, format='png')
    #    print(f"SHAP dependence plot saved at: {save_path}")
    return explainer, X_val_original, shap_values_val


# Updated function for SHAP with time interpolation
def plot_shap_values_for_deepsurv(model, X_train, X_val, scaler, cols_x, times):
    """
    Compute and plot SHAP values for PyCox DeepSurv models with time interpolation.

    Args:
        model: Trained PyCox DeepSurv model.
        X_train (pd.DataFrame): Scaled training feature data.
        X_val (pd.DataFrame): Scaled validation feature data.
        scaler: Fitted scaler used for preprocessing features.
        cols_x (list): List of feature column names.
        times (list or np.ndarray): Time points to aggregate survival probabilities.

    Returns:
        None
    """
    # Reverse scaling for SHAP interpretability
    print("Reversing scaling for interpretability...")
    X_train_original = reverse_scaling(X_train[cols_x], scaler, feature_names=cols_x)
    X_val_original = reverse_scaling(X_val[cols_x], scaler, feature_names=cols_x)

    # Wrap the PyCox model's prediction method for SHAP
    def model_predict(X):
        """
        Custom prediction function for SHAP.
        Predicts interpolated survival probabilities at specified time points.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)  # Convert to PyTorch tensor
        survival_preds = model.predict_surv_df(X_tensor)  # Predict survival curves

        # Interpolate survival probabilities at the specified times
        interpolated_probs = survival_preds.reindex(
            survival_preds.index.union(times)
        ).interpolate(method="index").loc[times]

        # Average over time points for SHAP
        mean_probs = interpolated_probs.mean(axis=0)
        return mean_probs.values

    # Initialize SHAP KernelExplainer
    print("Initializing SHAP KernelExplainer...")
    explainer = shap.Explainer(model_predict, X_train_original.values)

    # Compute SHAP values for the validation dataset
    print("Computing SHAP values for validation dataset...")
    shap_values_val = explainer(X_val_original.values)

    # Plot SHAP waterfall plot for the first validation sample
    print("Generating SHAP waterfall plot for the first validation sample...")
    shap.waterfall_plot(shap_values_val[0], feature_names=cols_x)
    plt.show()

    # Plot SHAP summary plot for the entire validation dataset
    print("Generating SHAP summary plot for the validation dataset...")
    #shap.summary_plot(shap_values_val, features=X_val_original, feature_names=cols_x)

    # Plot SHAP dependence plot for the most important feature
    top_feature = cols_x[np.argmax(abs(shap_values_val.values).mean(axis=0))]
    print(f"Generating SHAP dependence plot for the top feature: {top_feature}")
    #shap.dependence_plot(top_feature, shap_values_val.values, X_val_original, feature_names=cols_x)


def plot_patient_risk_scores(models, X_test, patient_ids, cols_x, name, times=np.arange(1, 20), color_list=None, title_suffix="Survival Curve", y_limits=(0, 1), save_folder=None):
    """
    Plot risk scores over time for selected patients using model predictions.

    Args:
        models (dict): Dictionary of trained models.
        X_test (pd.DataFrame): Test dataset containing feature data.
        patient_ids (list): List of patient IDs to plot.
        cols_x (list): List of feature column names.
        name (str): Name of the model to use for predictions.
        times (np.ndarray): Time points for evaluating survival probabilities.
        color_list (list, optional): List of colors for plotting. Defaults to a predefined color list.
        title_suffix (str, optional): Suffix for the plot title. Defaults to "Survival Curve".
        y_limits (tuple, optional): Tuple specifying the limits of the y-axis. Defaults to (0, 1).
        save_folder (str, optional): Folder to save the plot as a .png file.

    Returns:
        None
    """
    # Define a default color list if none is provided
    if color_list is None:
        color_list = [
            "#11cd9a", "#f0614a", "#ab63fa", "#ef553b", "#1f77b4",
            "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]

    # Ensure the specified model exists
    if name not in models:
        raise ValueError(f"Model '{name}' not found in the provided models dictionary.")

    # Generate survival probabilities
    model = models[name]
    survs = model.predict_surv_df(np.array(X_test[cols_x]).astype(np.float32))

    # Create the plot
    plt.figure(figsize=(10, 5))

    # Plot each patient's survival curve with assigned colors
    for idx, patient_id in enumerate(patient_ids):
        if patient_id not in X_test.index:
            print(f"Warning: Patient ID {patient_id} not found in X_test.")
            continue

        color = color_list[idx % len(color_list)]  # Cycle through colors if more patients than colors
        plt.plot(
            survs.index,  # X-axis: Time in years
            survs.iloc[:, patient_id],  # Y-axis: Survival probability for the patient
            label=f"Patient {patient_id}",
            color=color
        )

    # Customize the plot
    plt.title(f"{title_suffix} ({name})", fontsize=14)
    plt.xlabel('Time (years)', fontsize=12)
    plt.ylabel('Survival Probability (%)', fontsize=12)
    plt.ylim(*y_limits)

    # Format the y-axis to display percentages
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))

    # Customize legend
    legend = plt.legend()
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    # Add grid and layout adjustments
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot if save_folder is provided
    if save_folder:
        save_path = f"{save_folder}/results/patient_risk_scores.png"
        plt.savefig(save_path, format='png')

    # Show the plot
    plt.show()

def plot_shap_values_for_deepsurv2(model, X_train, y_train, X_val, scaler, cols_x, save_folder=None):
    import shap
    import matplotlib.pyplot as plt
    import numpy as np

    # Reverse scaling for SHAP interpretation
    X_train_original = reverse_scaling(X_train[cols_x], scaler, feature_names=cols_x)
    X_val_original = reverse_scaling(X_val[cols_x], scaler, feature_names=cols_x)

    # Initialize SHAP Explainer for DeepSurv
    print("Initializing SHAP explainer for DeepSurv...")
    times = [10]
    #def model_predict(X):
    #    """Predict risk scores using the DeepSurv model."""
    #    return model.predict(X)
    def model_predict(X):
        """
        Custom prediction function for SHAP.
        Predicts interpolated survival probabilities at specified time points.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # Convert to PyTorch tensor
        survival_preds = model.predict_surv_df(X_tensor)  # Predict survival curves

        # Interpolate survival probabilities at the specified times
        interpolated_probs = survival_preds.reindex(
            survival_preds.index.union(times)
        ).interpolate(method="index").loc[times]

        # Average over time points for SHAP
        mean_probs = interpolated_probs.mean(axis=0)
        return mean_probs.values

    explainer = shap.Explainer(model_predict, X_train_original.values)

    # Compute SHAP values for validation dataset
    print("Computing SHAP values for the validation dataset...")
    shap_values_val = explainer(X_val_original)

    # Plot SHAP waterfall plot for the first validation sample
    print("Generating SHAP waterfall plot for the first validation sample...")
    shap.plots.waterfall(shap_values_val[0])
    if save_folder:
        save_path = f"{save_folder}/results/shap_waterfall.png"
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=200)
        print(f"SHAP waterfall plot saved at: {save_path}")

    # Plot SHAP summary plot for validation dataset
    print("Generating SHAP summary plot for validation dataset...")
    shap.summary_plot(shap_values_val, X_val_original)
    if save_folder:
        save_path = f"{save_folder}/results/shap_summary.png"
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=200)
        print(f"SHAP summary plot saved at: {save_path}")

    # Plot SHAP dependence plot for the most important feature
    top_feature = cols_x[np.argmax(np.abs(shap_values_val.values).mean(axis=0))]
    print(f"Generating SHAP dependence plot for the top feature: {top_feature}")
    shap.dependence_plot(top_feature, shap_values_val.values, X_val_original)
    if save_folder:
        save_path = f"{save_folder}/results/shap_dependence.png"
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=200)
        print(f"SHAP dependence plot saved at: {save_path}")

    return explainer, X_val_original, shap_values_val