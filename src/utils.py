# utils.py
import torch
import numpy as np
import pandas as pd
import random
import os
import warnings

# --- Sklearn Metrics (needed for compute_metrics) ---
# Only import what's needed by functions remaining in this file
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix) # confusion_matrix needed for specificity

# --- General Utilities ---

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

# Suppress FutureWarning related to sklearn potentially
warnings.filterwarnings('ignore', category=FutureWarning)

def compute_iqr_threshold(data):
    """
    Calculates threshold based on IQR of the 'pearson' column.

    WARNING: This function requires a 'pearson' column in the input DataFrame,
             which is NOT produced by the current test_model function in test.py.
    """
    if 'pearson' not in data.columns:
        raise ValueError("Evaluation Error: DataFrame must contain a 'pearson' column for compute_iqr_threshold.")

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
    Computes various classification metrics based on the 'pearson' column and threshold.
    (Does not perform plotting).

    WARNING: This function requires 'pearson' and 'label' columns in the input DataFrame,
             The 'pearson' column is NOT produced by the current test_model function in test.py.
    """
    if 'pearson' not in data.columns or 'label' not in data.columns:
        raise ValueError("Evaluation Error: DataFrame must contain 'pearson' and 'label' columns for compute_metrics.")

    valid_data = data.dropna(subset=['pearson'])
    if valid_data.empty:
        print("Error: No valid data points remaining for metrics calculation. Returning zero/NaN metrics.")
        return {'Accuracy': 0, 'Precision': np.nan, 'Recall': np.nan, 'F1 Score': np.nan, 'Specificity': np.nan}

    predictions = (valid_data['pearson'] >= threshold).astype(int)
    labels = valid_data['label']

    if len(np.unique(labels)) < 2:
        print(f"Warning: Only one class ({np.unique(labels)}) present in true labels for threshold {threshold}. Metrics may be invalid.")
        accuracy = accuracy_score(labels, predictions)
        return {'Accuracy': accuracy, 'Precision': np.nan, 'Recall': np.nan, 'F1 Score': np.nan, 'Specificity': np.nan}

    # Calculate metrics, handling potential division by zero
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0) # zero_division=0 returns 0
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    # Specificity requires confusion matrix elements
    try:
        cm = confusion_matrix(labels, predictions, labels=[0, 1]) # Ensure labels=[0,1] for TN,FP,FN,TP order
        if cm.shape == (2,2):
            TN, FP, FN, TP = cm.ravel()
            specificity = TN / (TN + FP) if (TN + FP) != 0 else 0 # Handle division by zero
        else:
            # This case might happen if predictions contain only one class.
            # We need to infer specificity based on what was predicted.
            unique_preds = np.unique(predictions)
            if len(unique_preds) == 1:
                if unique_preds[0] == 0: # Only predicted Negative (TN + FN exist)
                    # We can calculate TNR = TN / (TN + FP). If FP is 0 because no positives were predicted, TNR = TN / TN = 1 (if TN > 0)
                    # Calculate TN from confusion matrix (might be 1x1 or 1x2 etc. if cm wasn't forced to 2x2)
                    tn_calc = sum((labels == 0) & (predictions == 0))
                    fp_calc = sum((labels == 0) & (predictions == 1)) # Should be 0
                    specificity = tn_calc / (tn_calc + fp_calc) if (tn_calc + fp_calc) != 0 else 0
                else: # Only predicted Positive (FP + TP exist), TN must be 0
                    specificity = 0.0
            else: # Unexpected shape
                print(f"Warning: Confusion matrix shape {cm.shape} unexpected. Cannot reliably calculate specificity.")
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