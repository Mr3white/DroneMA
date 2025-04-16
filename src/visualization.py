# visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings

# --- Configuration (Optional, if needed directly) ---
# from config import config # Uncomment if config is directly needed here

# --- Sklearn for metrics/plotting ---
from sklearn.metrics import (roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay)

# --- Import non-plotting evaluation helpers from utils ---
# Assuming utils.py is in the same directory or accessible via PYTHONPATH
try:
    from utils import compute_iqr_threshold, compute_metrics
except ImportError:
    print("Error: Could not import 'compute_iqr_threshold' and 'compute_metrics' from utils.py.")
    print("Ensure utils.py is in the correct path and contains these functions.")
    # Define dummy functions to avoid crashing if import fails, but functionality will be lost
    def compute_iqr_threshold(data): print("DUMMY: compute_iqr_threshold"); return 0
    def compute_metrics(data, threshold): print("DUMMY: compute_metrics"); return {}


# Suppress FutureWarning related to sklearn potentially
warnings.filterwarnings('ignore', category=FutureWarning)

# === Functions originally provided by user ===

# def plot_losses(train_losses, test_losses, save_path=config.results_save_path + 'losses.png'):
# Modified to not depend directly on config import if possible
def plot_losses(train_losses, test_losses, save_path=None):
    """Plots training and validation/test losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Validation/Test Loss') # More general label
    plt.title('Training and Validation/Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        try:
            plt.savefig(save_path)
            print(f"Loss plot saved to {save_path}")
        except Exception as e:
             print(f"Error saving loss plot to {save_path}: {e}")
    # plt.show() # Optional: uncomment to display plot interactively
    plt.close() # Close plot to free memory

# def plot_predictions(predictions, targets, save_path=config.results_save_path + 'predictions.png'):
# Modified to not depend directly on config import if possible
def plot_predictions(predictions, targets, save_path=None):
    """Plots model predictions against ground truth."""
    plt.figure(figsize=(12, 6))
    # Ensure inputs are numpy arrays for plotting
    if isinstance(predictions, torch.Tensor): predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor): targets = targets.detach().cpu().numpy()

    plt.plot(predictions.flatten(), label='Predictions', alpha=0.7) # Flatten in case of extra dims
    plt.plot(targets.flatten(), label='Ground Truth', alpha=0.7) # Flatten
    plt.title('Predictions vs Ground Truth')
    plt.xlabel('Sample Index / Time Steps') # More general label
    plt.ylabel('Value')
    plt.legend()
    if save_path:
        try:
            plt.savefig(save_path)
            print(f"Prediction plot saved to {save_path}")
        except Exception as e:
             print(f"Error saving prediction plot to {save_path}: {e}")
    # plt.show() # Optional: uncomment to display plot interactively
    plt.close()


# === Functions moved from utils.py ===

def compute_confusion_matrix(data, threshold, save_path_file):
    """
    Computes, plots, and saves the confusion matrix based on 'pearson' column.

    WARNING: This function requires 'pearson' and 'label' columns in the input DataFrame,
             The 'pearson' column is NOT produced by the current test_model function in test.py.
    """
    if 'pearson' not in data.columns or 'label' not in data.columns:
        raise ValueError("Evaluation Error: DataFrame must contain 'pearson' and 'label' columns for compute_confusion_matrix.")

    valid_data = data.dropna(subset=['pearson'])
    if valid_data.empty:
        print("Error: No valid data points remaining for confusion matrix calculation.")
        return

    predictions = (valid_data['pearson'] >= threshold).astype(int)
    labels = valid_data['label']

    if len(np.unique(labels)) < 2 :
         print(f"Warning: Only one class present in true labels for confusion matrix (Threshold={threshold:.3f}). Plot may be misleading.")
    elif len(np.unique(predictions)) < 2 :
         print(f"Warning: Only one class predicted for confusion matrix (Threshold={threshold:.3f}). Plot may be misleading.")

    try:
        # Ensure labels are correctly inferred or provided if necessary
        unique_labels_in_data = sorted(labels.unique()) # Get actual unique labels
        cm = confusion_matrix(labels, predictions, labels=unique_labels_in_data) # Explicitly pass labels
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels_in_data)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix (Threshold={threshold:.3f})')
        plt.savefig(save_path_file)
        plt.close()
        print(f"    Confusion matrix saved to {save_path_file}")
    except Exception as e:
        print(f"Error calculating or plotting confusion matrix: {e}")


def plot_and_evaluate(data_path_pos, data_path_neg, save_path_plots_metrics):
    """
    Loads data, calculates threshold, plots ROC/Confusion Matrix, computes metrics.

    WARNING: This function expects input CSVs ('pos', 'neg') to contain 'pearson'
             and 'label' columns. The 'pearson' column is NOT produced by the
             current test_model function in test.py. This function WILL likely fail.
    """
    print(f"  Attempting evaluation using {data_path_pos} and {data_path_neg}")
    try:
        pos_data = pd.read_csv(data_path_pos)
        neg_data = pd.read_csv(data_path_neg)
        print(f"    Successfully loaded CSV files.")
    except FileNotFoundError as e:
        print(f"    Error loading data files for evaluation: {e}. Skipping evaluation.")
        return None
    except Exception as e:
         print(f"    Error reading CSV files: {e}")
         return None

    # --- Data Preparation ---
    pos_data['label'] = 1
    neg_data['label'] = 0
    data = pd.concat([pos_data, neg_data], ignore_index=True)

    # Check for 'pearson' column existence EARLY
    if 'pearson' not in data.columns:
        print("    ERROR: Evaluation failed. Required 'pearson' column not found in loaded data.")
        print(f"    Columns found: {data.columns.tolist()}")
        print(f"    Please check the output format of the 'test_model' function in test.py.")
        return None # Stop evaluation if the crucial column is missing

    initial_rows = len(data)
    data = data.dropna(subset=['pearson'])
    if len(data) < initial_rows:
        print(f"    Warning: Removed {initial_rows - len(data)} rows with NaN 'pearson' scores.")

    if data.empty or data['pearson'].isnull().all():
         print("    Error: No valid 'pearson' data available after loading/cleaning. Skipping evaluation.")
         return None
    if len(data['label'].unique()) < 2:
         print("    Error: Data contains only one class label after loading. ROC/AUC invalid. Skipping.")
         return None

    # --- Calculate IQR Threshold (Imported from utils) ---
    print("    Calculating IQR threshold...")
    try:
        threshold = compute_iqr_threshold(data) # Calls function from utils
        print(f"    IQR Threshold determined: {threshold:.5f}")
    except Exception as e:
         print(f"    Error computing IQR threshold: {e}")
         return None

    # --- ROC Curve Plotting (Here in visualization.py) ---
    print("    Calculating and plotting ROC curve...")
    labels = data['label'].values
    scores = data['pearson'].values
    try:
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
    except Exception as e:
         print(f"    Error calculating ROC curve/AUC: {e}")
         return None # Cannot proceed without ROC

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:0.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.5)
    roc_save_path = os.path.join(save_path_plots_metrics, 'roc_curve.png')
    try:
        plt.savefig(roc_save_path)
        print(f"    ROC curve saved to {roc_save_path}")
    except Exception as e:
         print(f"    Error saving ROC curve plot: {e}")
    plt.close()


    # --- Confusion Matrix Plotting (Here in visualization.py) ---
    print(f"    Calculating confusion matrix using IQR threshold ({threshold:.5f})...")
    cm_save_path = os.path.join(save_path_plots_metrics, 'confusion_matrix_iqr.png')
    try:
        # Calls the compute_confusion_matrix defined above in this file
        compute_confusion_matrix(data, threshold, cm_save_path)
    except Exception as e:
         print(f"    Error computing/saving confusion matrix: {e}")
         # Continue to metrics calculation if possible, or return None

    # --- Metrics Calculation (Imported from utils) ---
    print(f"    Calculating metrics using IQR threshold ({threshold:.5f})...")
    try:
        metrics = compute_metrics(data, threshold) # Calls function from utils
        print(f"    Metrics calculated: {metrics}")
    except Exception as e:
         print(f"    Error computing metrics: {e}")
         return None # If metrics fail, return None

    # Add AUC and threshold to the metrics dict
    metrics['AUC'] = roc_auc
    metrics['Threshold_IQR'] = threshold

    return metrics