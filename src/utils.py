# utils.py
import torch
import numpy as np
import pandas as pd
import random
import os
import warnings
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)

# Assuming these are correctly placed and imported by main
# from data_processing import create_eval_dataloader
# from test import test_model


# --- General Utilities ---
# seed_everything, get_files, create_dirs, normalize
# (Keep these functions as they were in the previous response)
def seed_everything(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_files(folder_path, extension='csv'):
    """Recursively gets files with a specific extension from a folder."""
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list

def create_dirs(*paths):
    """Creates directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)

def normalize(tensor, eps=1e-8):
    """Performs window-level normalization on a tensor."""
    if tensor.dim() != 3:
         print(f"Warning: normalize function expected tensor with 3 dims (batch, seq, feat), got {tensor.dim()}. Normalization might be incorrect.")
         return tensor
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True)
    return (tensor - mean) / (std + eps)


# --- Evaluation Utilities (Non-Plotting) ---

warnings.filterwarnings('ignore', category=FutureWarning)

def compute_iqr_threshold(data):
    """
    Calculates threshold based on IQR of the 'pearson' column.
    Expects a DataFrame with a 'pearson' column.
    """
    if 'pearson' not in data.columns:
        raise ValueError("DataFrame must contain a 'pearson' column for compute_iqr_threshold.")

    pearson_data = data['pearson'].dropna()
    if pearson_data.empty:
         print("Warning: 'pearson' column is empty or all NaN after dropping NaNs. Returning threshold 0.")
         return 0

    Q1 = pearson_data.quantile(0.25)
    Q3 = pearson_data.quantile(0.75)
    IQR = Q3 - Q1
    threshold = Q1 - 1.5 * IQR
    threshold = max(threshold, 0) # Cap threshold at 0
    return threshold


def compute_metrics(data, threshold):
    """
    Computes classification metrics based on the 'pearson' column and a *given* threshold.
    (Does not perform plotting).
    """
    if 'pearson' not in data.columns or 'label' not in data.columns:
        raise ValueError("Evaluation Error: DataFrame must contain 'pearson' and 'label' columns for compute_metrics.")

    valid_data = data.dropna(subset=['pearson'])
    if valid_data.empty:
        print("Error: No valid data points remaining for metrics calculation. Returning zero/NaN metrics.")
        return {'Accuracy': 0, 'Precision': np.nan, 'Recall': np.nan, 'F1 Score': np.nan, 'Specificity': np.nan}

    # Apply the *provided* threshold
    predictions = (valid_data['pearson'] >= threshold).astype(int)
    labels = valid_data['label']

    # --- Calculation logic remains the same as before ---
    if len(np.unique(labels)) < 2:
        print(f"Warning: Only one class ({np.unique(labels)}) present in true labels for threshold {threshold}. Metrics may be invalid.")
        accuracy = accuracy_score(labels, predictions)
        return {'Accuracy': accuracy, 'Precision': np.nan, 'Recall': np.nan, 'F1 Score': np.nan, 'Specificity': np.nan}

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    try:
        cm = confusion_matrix(labels, predictions, labels=[0, 1])
        if cm.shape == (2,2):
            TN, FP, FN, TP = cm.ravel()
            specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        else:
             unique_preds = np.unique(predictions)
             if len(unique_preds) == 1:
                 tn_calc = sum((labels == 0) & (predictions == 0))
                 fp_calc = sum((labels == 0) & (predictions == 1))
                 specificity = tn_calc / (tn_calc + fp_calc) if (tn_calc + fp_calc) != 0 else (1.0 if unique_preds[0] == 0 else 0.0) # Adjusted logic
             else:
                 print(f"Warning: Confusion matrix shape {cm.shape} unexpected. Cannot calculate specificity.")
                 specificity = np.nan
    except Exception as e:
        print(f"Error calculating confusion matrix/specificity: {e}")
        specificity = np.nan

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Specificity': specificity
    }


# --- NEW FUNCTION ---
def calculate_thresholds_from_train_data(train_files, window_list, model,
                                         create_dataloader_func, test_func,
                                         device, loss_fn, ew_alpha,
                                         temp_path="temp_train_eval"):
    """
    Calculates IQR thresholds based on model performance on training data
    for different window sizes.

    Args:
        train_files (list): List of file paths for the training data.
        window_list (list): List of window sizes (e.g., [10, 20, ...]).
        model: The trained model.
        create_dataloader_func: Function to create a dataloader (e.g., create_eval_dataloader).
        test_func: Function to run model inference and save results (e.g., test_model).
                   Must save a 'pearson' column.
        device: Torch device.
        loss_fn: Loss function (needed by test_func).
        ew_alpha: EW Alpha value (needed by test_func).
        temp_path (str): Directory to store temporary CSV files.

    Returns:
        dict: A dictionary mapping window size to the calculated IQR threshold.
    """
    print("\nCalculating IQR thresholds from training data...")
    thresholds = {}
    model.eval() # Ensure model is in eval mode
    os.makedirs(temp_path, exist_ok=True) # Create temp dir

    for window_size in window_list:
        print(f"  Processing training data for window_size = {window_size}...")
        temp_csv_path = os.path.join(temp_path, f"train_results_window_{window_size}.csv")

        # 1. Create DataLoader for training files with current window size
        try:
            # Use eval dataloader (shuffle=False), batch size can be adjusted
            train_eval_loader = create_dataloader_func(train_files, window_size, batch_size=1) # Or other suitable batch size
        except Exception as e:
            print(f"    Error creating dataloader for training files (window {window_size}): {e}")
            thresholds[window_size] = 0 # Assign default threshold on error
            continue

        # 2. Run inference using the test_func to generate results CSV
        if len(train_eval_loader.dataset) > 0:
            try:
                test_func(model, train_eval_loader, device, loss_fn, save_path=temp_csv_path, ew_alpha=ew_alpha)
            except Exception as e:
                print(f"    Error running test_func on training data (window {window_size}): {e}")
                thresholds[window_size] = 0 # Assign default threshold on error
                continue # Skip to next window size
        else:
            print(f"    Skipping threshold calculation for window {window_size}: No data after preprocessing training files.")
            thresholds[window_size] = 0 # Assign default threshold
            continue

        # 3. Load results and calculate IQR threshold
        try:
            if os.path.exists(temp_csv_path):
                train_results_df = pd.read_csv(temp_csv_path)
                if 'pearson' in train_results_df.columns and not train_results_df['pearson'].isnull().all():
                     # Calculate threshold using the existing utility function
                     threshold = compute_iqr_threshold(train_results_df)
                     thresholds[window_size] = threshold
                     print(f"    Calculated threshold for window {window_size}: {threshold:.6f}")
                else:
                     print(f"    Warning: 'pearson' column missing or all NaN in {temp_csv_path}. Setting threshold to 0.")
                     thresholds[window_size] = 0
                # 4. Optional: Clean up temporary file
                # os.remove(temp_csv_path)
            else:
                 print(f"    Warning: Temp results file not found: {temp_csv_path}. Setting threshold to 0.")
                 thresholds[window_size] = 0

        except Exception as e:
            print(f"    Error loading results or calculating threshold for window {window_size}: {e}")
            thresholds[window_size] = 0 # Assign default on error

    # Optional: Remove temp directory if empty or desired
    # try:
    #     if not os.listdir(temp_path):
    #         os.rmdir(temp_path)
    # except OSError:
    #     print(f"Could not remove temporary directory: {temp_path}")

    print("Finished calculating thresholds from training data.")
    return thresholds