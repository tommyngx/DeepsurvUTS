import itertools
import pandas as pd
import numpy as np
import datetime
import time
import os

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

from sksurv.functions import StepFunction
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import integrated_brier_score, brier_score

import matplotlib.pyplot as plt
import requests
from matplotlib import font_manager

def plot_feat_imp(feature_names, feature_importances, top_n=10):
    """
    Plots the top_n feature importances.

    Parameters:
    - feature_names: list of strings, shape (n_features,)
    - feature_importances: array-like, shape (n_features,)
    - top_n: int, number of top features to plot
    """
    # Apply ggplot style
    plt.style.use('ggplot')

    # Download the font file
    font_url = 'https://github.com/tommyngx/style/blob/main/Poppins.ttf?raw=true'
    font_path = 'Poppins.ttf'
    response = requests.get(font_url)
    with open(font_path, 'wb') as f:
        f.write(response.content)

    # Load the font
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    # Convert feature_importances to a NumPy array
    feature_importances = np.array(feature_importances)

    # Get the indices of the top_n feature importances
    indices = feature_importances.argsort()[-top_n:][::-1]

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), feature_importances[indices], color='#8f63f4', align='center')
    # Capitalize the first letter of each feature name
    capitalized_feature_names = [feature_names[i].capitalize() for i in indices]
    plt.yticks(range(len(indices)), capitalized_feature_names, fontproperties=prop)
    plt.xlabel('Relative Importance', fontproperties=prop, fontsize=14)
    plt.ylabel('Features', fontproperties=prop, fontsize=14)
    plt.title('Top {} Feature Importances'.format(top_n), fontproperties=prop, fontsize=16)
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top
    plt.tight_layout()
    plt.show()
    return indices
    
    
def plot_feat_imp_ori(cols, coef):
    feat_importance = pd.DataFrame({
        "feature": cols,
        "coef": coef
    })
    feat_importance["coef_abs"] = abs(feat_importance.coef)
    feat_importance.sort_values(by='coef_abs', ascending=True, inplace=True)

    fig = px.bar(feat_importance, x="coef", y="feature", height= 500, width= 600)
    
    fig.update_layout(
        dict(
            xaxis={'title' : 'Coefficient'}, 
            yaxis={'title' : 'Feature'}
        )
    )
    
    return feat_importance, fig
    
def get_bier_score(df, y_train, y_test, survs, times, col_target = "duration", with_benchmark=True):
    
    if with_benchmark:
    
        km_func = StepFunction(
            *kaplan_meier_estimator(df["censored"].astype(bool), df[col_target])
        )
        
        preds = {
            'estimator': np.row_stack([ fn(times) for fn in survs]),
            'random': 0.5 * np.ones((df.shape[0], times.shape[0])),
            'kaplan_meier': np.tile(km_func(times), (df.shape[0], 1))
        }
        
    else:
        preds = {'estimator': np.row_stack([ fn(times) for fn in survs])}
        
    scores = {}
    for k, v in preds.items():
        scores[k] = integrated_brier_score(y_train, y_test, v, times)
    
    return scores



def get_bier_curve(y_train, y_test, survs, times):
    preds = {'estimator': np.row_stack([fn(times) for fn in survs])}

    scores = []
    for t in times:
        preds = [fn(t) for fn in survs]
        _, score = brier_score(y_train, y_test, preds, t)
        scores.append(score[0])

    return scores
    

def fit_score(estimator, Xy, train_index, test_index, cols, col_target):
    Xy_train = Xy.loc[train_index]
    Xy_test = Xy.loc[test_index]

    y_train = np.array(
        list(zip(Xy_train.censored, Xy_train[col_target])),
        dtype=[('censored', '?'), (col_target, '<f8')])

    y_test = np.array(
        list(zip(Xy_test.censored, Xy_test[col_target])),
        dtype=[('censored', '?'), (col_target, '<f8')])

    estimator = estimator.fit(Xy_train[cols], y_train)

    score = estimator.score(Xy_test[cols], y_test)
    
    return estimator, score   


def cv_fit_score(df, cv, estimator_fn, cols, col_target, params, drop_zero = True, verbose = False):
    
    Xy = df[cols+["censored", col_target]].dropna().reset_index(drop=True)
    
    if drop_zero:
        index_z = Xy[Xy[col_target]==0].index
        Xy = Xy.drop(index_z, axis=0).reset_index(drop=True)
    
    y = list(zip(Xy.censored, Xy[col_target]))
    y = np.array(y, dtype=[('censored', '?'), (col_target, '<f8')])
    
    cv_scores = {}

    t0 = time.time()
    for i, (train_index, test_index) in enumerate(cv.split(Xy)):

        estimator = estimator_fn(**params)
        estimator, score = fit_score(estimator, Xy, train_index, test_index, cols, col_target)

        if verbose:
            print(f"Fold {i}: {round(score, 3)}")

        cv_scores["fold_"+str(i)] = score
    
    
    cv_scores["time"] = (time.time() - t0)/60
    
    return estimator, cv_scores


def grid_search(grid_params, df, cv, estimator_fn, cols, col_target, verbose = False):
    
    best_score = -100
    
    n = 1
    for k, v in grid_params.items():
        n*=len(v)
        
    print(f'{n} total scenario to run')
    
    try: 
    
        for i, combi in enumerate(itertools.product(*grid_params.values())):
            params = {k:v for k,v in zip(grid_params.keys(), combi)}
            
            print(f'{i+1}/{n}: params: {params}')
            
            estimator, cv_scores = cv_fit_score(df, cv, estimator_fn, cols, col_target, params, verbose = verbose)
            
            table = pd.DataFrame.from_dict(cv_scores, orient='index').T
            cols_fold = [c for c in table.columns if 'fold' in c]
            table['mean'] = table[cols_fold].mean(axis=1)
            table['std'] = table[cols_fold].std(axis=1)
    
            for k, v in params.items():
                table[k] = v
    
            table = table[list(params.keys()) + [c for c in table.columns if c not in params]]
        
            results = table if i==0 else pd.concat([results, table], axis=0)
    
            if best_score < table['mean'].iloc[0]:
                best_score = table['mean'].iloc[0]
                best_estimator = estimator
    
    except KeyboardInterrupt:
        pass

    return best_estimator, results.reset_index(drop=True)


def train_and_save_ML_models(estimators, train_x, train_y, test_x, test_y, cols_x, save_model):
    """
    Train a list of models, evaluate their performance, and save them to a specified folder.

    Args:
        estimators (dict): A dictionary of model names and their corresponding estimators.
        train_x (pd.DataFrame): Training features.
        train_y (pd.Series): Training target values.
        test_x (pd.DataFrame): Testing features.
        test_y (pd.Series): Testing target values.
        cols_x (list): List of feature column names.
        save_model (str): Path to the folder where models should be saved.

    Returns:
        pd.DataFrame: A DataFrame containing training and testing scores for each model.
    """
    # Ensure the save folder and models subfolder exist
    save_model2 = f"{save_model}/models"
    os.makedirs(save_model2, exist_ok=True)

    # Create a results folder
    results_folder = f"{save_model}/results"
    os.makedirs(results_folder, exist_ok=True)

    scores = {}  # Dictionary to store training and testing scores

    for name, estimator in estimators.items():
        print(f"Training model: {name}")
        # Train the model
        estimator.fit(train_x[cols_x], train_y)

        # Evaluate the model
        scores[name] = {
            'score_train': estimator.score(train_x[cols_x], train_y),
            'score_test': estimator.score(test_x[cols_x], test_y)
        }

        # Save the trained model
        model_path = os.path.join(save_model2, f"{name}_ML.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(estimator, f)
        print(f"Model saved at: {model_path}")

        # If the model is cox_ph, plot feature importance
        if name == "cox_ph" and hasattr(estimator, "coef_"):
            feat_importance_cox = plot_feat_imp(cols_x, estimator.coef_)
            print(f"Feature importance plotted for {name}.")

    # Convert scores dictionary to a DataFrame
    scores_df = pd.DataFrame.from_dict(scores, orient='index').reset_index()
    scores_df.rename(columns={'index': 'model'}, inplace=True)

    # Round scores to 5 decimal places
    scores_df[['score_train', 'score_test']] = scores_df[['score_train', 'score_test']].round(5)

    print("Model scores:\n", scores_df)

    # Save the scores DataFrame as a CSV in the results folder
    results_csv_path = os.path.join(results_folder, "result.csv")
    scores_df.to_csv(results_csv_path, index=False)
    print(f"Scores saved at: {results_csv_path}")

    return scores_df