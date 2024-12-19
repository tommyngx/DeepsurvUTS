import pandas as pd
import os
import pickle
#from sklearn.model_selection import train_test_split, split_train_test
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from pycox.models import DeepHitSingle, CoxPH
from sklearn.model_selection import train_test_split
import numpy as np
import torchtuples as tt
import json
import warnings, requests

def getGoogleSeet(spreadsheet_id, outDir, outFile):

  url = f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv'
  response = requests.get(url)
  if response.status_code == 200:
    filepath = os.path.join(outDir, outFile)
    with open(filepath, 'wb') as f:
      f.write(response.content)
      print('CSV file saved to: {}'.format(filepath))
  else:
    print(f'Error downloading Google Sheet: {response.status_code}')


def extract_survival_data(csv_path, dataset_name, cols_x, col_target):
    """
    Process a survival analysis dataset from a CSV file, check for missing values,
    remove rows with missing values, and filter columns based on user input.

    Args:
        csv_path (str): Path to the CSV file.
        dataset_name (str): Name of the dataset to filter on ('SOF' or 'MrOS').
        cols_x (list): List of selected feature columns.
        col_target (list): List containing target columns ['event', 'time_to_event'].

    Returns:
        pd.DataFrame: Processed and filtered DataFrame with selected columns.
    """
    # Load the data
    df = pd.read_csv(csv_path)

    # Print initial number of rows
    #print(f"Initial number of rows: {len(df)}")

    # Map 'gender' column: 'F' -> 1, 'M' -> 0
    gender_mapping = {'F': 1, 'M': 0}
    df['gender'] = df['gender'].map(gender_mapping)

    # Filter the DataFrame based on the specified dataset
    df = df[df['data'] == dataset_name]

    # Add 'censored' column: True if event == 0, else False
    df['censored'] = df[col_target[0]].apply(lambda x: True if x == 0 else False)

    # Rename or create time-to-event column
    df['time2event'] = df[col_target[1]]

    # Filter the DataFrame to include only selected columns
    selected_columns = cols_x + [col_target[0], 'censored', 'time2event']
    df = df[selected_columns]
    print("Columns in the dataset:", df.columns)

    # Check for missing values
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    if not missing_summary.empty:
        print("Missing values detected:\n", missing_summary)
        print(f"Number of rows before removing missing values: {len(df)}")
        df = df.dropna()  # Remove rows with missing values
        print(f"Number of rows after removing missing values: {len(df)}")
    else:
        print("No missing values detected in the dataset.")

    # Print summary statistics
    print("Censored value counts:", df['censored'].value_counts())
    df = df.reset_index(drop=True)
    return df

def split_train_test(df, cols_x, col_target, test_size=0.3, col_stratify=None, random_state=None, dropna=True):
    Xy = df[cols_x+["censored", col_target]].dropna() if dropna else df[cols_x+["censored", col_target]]

    stratify = None if col_stratify is None else Xy[col_stratify]
    Xy_train, Xy_test = train_test_split(Xy, test_size=test_size, stratify=stratify, random_state=random_state)

    Xy_train.reset_index(drop=True, inplace=True)
    Xy_test.reset_index(drop=True, inplace=True)

    y_train = np.array(list(zip(Xy_train.censored, Xy_train[col_target])),
                       dtype=[('censored', '?'), (col_target, '<f8')])
    y_test = np.array(list(zip(Xy_test.censored, Xy_test[col_target])),
                       dtype=[('censored', '?'), (col_target, '<f8')])

    return Xy_train, Xy_test, y_train, y_test

def process_and_save_data(df, cols_x, col_target, random_state, scaler_name=None, oversample=None, save_data=None):
    """
    Process a dataset: split into train/test sets, apply scaling, oversampling, and save the results.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cols_x (list): List of feature columns.
        col_target (str): Target column to include.
        random_state (int): Random seed for reproducibility.
        scaler_name (str): Scaler to use ('MinMaxScaler' or None).
        oversample (bool): Whether to apply oversampling to the training data.
        save_data (str): Folder path to save processed data.

    Returns:
        dict: A dictionary containing train, validation, and test datasets (all mapped to test).
    """
    col_target= 'time2event'
    # Split into train/test
    Xy_train, Xy_test, y_train, y_test = split_train_test(
        df, cols_x, col_target, test_size=0.20, col_stratify= "censored", random_state=random_state
    )

    # Count rows
    n_train, n_test = Xy_train.shape[0], Xy_test.shape[0]
    n_tot = n_train + n_test
    print(f"Number of rows: Train={n_train}, Test={n_test}")
    print("Train: {}%, Test: {}%".format(
        round(n_train / n_tot * 100),
        round(n_test / n_tot * 100)
    ))

    # Apply scaling if specified
    scaler = None
    if scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler()
        Xy_train[cols_x] = scaler.fit_transform(Xy_train[cols_x])
        Xy_test[cols_x] = scaler.transform(Xy_test[cols_x])
        print("Scaling applied with MinMaxScaler.")

    # Oversample the training data if specified
    if oversample:
        Xy_train = oversample_dataframe(Xy_train, target_column="censored")
        print("Oversampling applied. Updated class distribution:")
        print(Xy_train["censored"].value_counts())

    # Save scaler if applicable
    if scaler and save_data:
        save_data = f"{save_data}/data"
        os.makedirs(save_data, exist_ok=True)
        with open(os.path.join(save_data, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        print("Scaler saved.")

    # Prepare data to save (map all splits to test for this use case)
    map_data = {
        'train_x': Xy_train[cols_x + ['censored', 'time2event']],
        #'val_x'  : Xy_test[cols_x + ['censored', 'time2event']],
        'test_x' : Xy_test[cols_x + ['censored', 'time2event']],
        'train_y': y_train,
        #'val_y'  : y_test,
        'test_y' : y_test,
    }

    # Save data to specified folder
    if save_data:
        for k, v in map_data.items():
            with open(os.path.join(save_data, f"{k}.pkl"), "wb") as f:
                pickle.dump(v, f)
        print(f"Processed data saved to {save_data}.")

    return map_data


def oversample_dataframe(df, target_column):
    """
    Oversample the minority class in a DataFrame based on the target column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the column to oversample.

    Returns:
        pd.DataFrame: A new DataFrame with oversampled data.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    return pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                      pd.DataFrame(y_resampled, columns=[target_column])], axis=1)


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


def generate_all_probabilities(models, X_test, y_time, y_censored, time_point, cols_x):
    """
    Generate a DataFrame of observed, predicted probabilities, and actual outcomes for all models.

    Args:
        models (dict): Dictionary containing the models ('deepsurv', 'deephit', 'cox_ph', 'svm', 'rsf', etc.).
        X_test (pd.DataFrame): Feature data for the test set.
        y_time (pd.Series): Time-to-event data for the test set.
        y_censored (pd.Series): Censoring indicator (1 = event, 0 = censored).
        time_point (float): Time point for which to compute probabilities.
        cols_x (list): List of feature columns used during model training.

    Returns:
        pd.DataFrame: DataFrame with patients, observed probabilities, model-predicted probabilities,
                      and actual outcomes (True = Event, False = No Event).
    """
    from lifelines import KaplanMeierFitter

    # Initialize Kaplan-Meier fitter
    kmf = KaplanMeierFitter()
    kmf.fit(y_time, y_censored)
    km_observed_probs = kmf.predict(time_point)

    # Create a DataFrame to store predicted probabilities
    predicted_probs = pd.DataFrame(index=X_test.index)
    predicted_probs['Observed Probability (Kaplan-Meier)'] = km_observed_probs

    # Add the actual outcome column with boolean values
    predicted_probs['Actual Outcome'] = y_censored.astype(int)

    # Compute predicted probabilities for each model
    for model_name, model in models.items():
        print(f"Processing model: {model_name}")
        X_test_filtered = X_test[cols_x]  # Only use columns used during training

        if model_name in ['gboost', 'cox_ph', 'rsf']:
            # Models with survival functions
            survival_function = model.predict_survival_function(X_test_filtered)
            predicted_probs[model_name] = [fn(time_point) for fn in survival_function]
        elif model_name == 'svm':
            # FastSurvivalSVM generates risk scores, convert them to probabilities
            risk_scores = model.predict(X_test_filtered)
            predicted_probs[model_name] = 1 / (1 + np.exp(-risk_scores))  # Sigmoid conversion
        elif model_name in ['deepsurv', 'deephit']:
            # DeepSurv and DeepHit specific computation
            survival_function = model.predict_surv_df(np.array(X_test_filtered).astype(np.float32))
            closest_time_point = survival_function.index.get_indexer([time_point], method="nearest")[0]
            predicted_probs[model_name] = survival_function.iloc[closest_time_point].values
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

    # Add a Patient column for unique identifiers
    predicted_probs.reset_index(inplace=True)
    predicted_probs.rename(columns={'index': 'Patient'}, inplace=True)

    # Add additional columns for time and outcome at the specified time point
    predicted_probs['time'] = y_time
    predicted_probs['outcomeTime'] = predicted_probs.apply(
        lambda row: row['Actual Outcome'] if row['time'] <= time_point else 0, axis=1
    )

    return predicted_probs