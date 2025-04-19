# test.py
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import os # Added for evaluate function
import matplotlib.pyplot as plt # Added for evaluate function
from sklearn.metrics import roc_curve, auc # Added for evaluate function

# --- Custom Module Imports ---
from utils import normalize, compute_metrics # compute_metrics is needed by evaluate
from visualization import compute_confusion_matrix # compute_confusion_matrix is needed by evaluate


def test_model(model, loader, device, loss_function, save_path='results.csv', ew_alpha=0):
    """
    Runs the model on the data from the loader, calculates metrics per batch,
    and saves detailed results per sample, including a 'pearson' column.
    """
    # ... (Content of test_model remains the same as the previous response) ...
    model.eval()
    results_list = []
    print(f"  Running test model, calculating metrics, saving results to {save_path}")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            try:
                inputs_normalized = normalize(inputs)
                targets_normalized = normalize(targets)
            except Exception as e: print(f"Error during normalization in batch {batch_idx}: {e}"); continue
            outputs = model(inputs_normalized)
            try:
                last_output = outputs[:, -1, :]
                last_target_norm = targets_normalized[:, -1, :]
                loss = loss_function(last_output, last_target_norm).item()
            except Exception as e: print(f"Error calculating loss in batch {batch_idx}: {e}"); loss = np.nan
            pearson_correlation = np.nan; spearman_correlation = np.nan
            try:
                predicted_flat = outputs.cpu().numpy().flatten()
                real_flat = targets_normalized.cpu().numpy().flatten()
                if len(predicted_flat) > 1 and len(real_flat) > 1 and len(predicted_flat) == len(real_flat) and \
                   np.std(predicted_flat) > 1e-6 and np.std(real_flat) > 1e-6:
                    if ew_alpha != 0: print("    Skipping EW correlation as apply_exponential_weights is missing.")
                    else:
                        pearson_correlation = pearsonr(predicted_flat, real_flat).statistic
                        spearman_correlation = spearmanr(predicted_flat, real_flat).statistic
                else: print(f"    Skipping correlation calculation for batch {batch_idx}: Invalid data shape or variance.")
            except Exception as e: print(f"Error calculating correlation in batch {batch_idx}: {e}")
            for i in range(batch_size):
                 results_list.append({'Batch_Index': batch_idx, 'Loss': loss, 'pearson': pearson_correlation, 'spearman': spearman_correlation,
                                      'input': inputs[i, -1, 0].item() if inputs.shape[-1] == 1 else inputs[i, -1, :].cpu().numpy(),
                                      'target': targets[i, -1, 0].item() if targets.shape[-1] == 1 else targets[i, -1, :].cpu().numpy(),
                                      'input_norm': inputs_normalized[i, -1, 0].item() if inputs_normalized.shape[-1] == 1 else inputs_normalized[i, -1, :].cpu().numpy(),
                                      'target_norm': targets_normalized[i, -1, 0].item() if targets_normalized.shape[-1] == 1 else targets_normalized[i, -1, :].cpu().numpy(),
                                      'output': outputs[i, -1, 0].item() if outputs.shape[-1] == 1 else outputs[i, -1, :].cpu().numpy()})
    if not results_list:
         print("Warning: No results collected. Output CSV will be empty."); results_df = pd.DataFrame()
    else:
        results_df = pd.DataFrame(results_list)
        cols = ['Batch_Index', 'Loss', 'pearson', 'spearman', 'input', 'target', 'input_norm', 'target_norm', 'output']
        cols = [c for c in cols if c in results_df.columns]; results_df = results_df[cols]
    try: results_df.to_csv(save_path, index=False, float_format='%.6f'); print(f"  Test results successfully saved to {save_path}")
    except Exception as e: print(f"Error saving test results to {save_path}: {e}")


# --- NEW FUNCTION (Moved from visualization.py and Renamed) ---
def evaluate(data_path_pos, data_path_neg, threshold, save_path_plots_metrics):
    """
    Loads inference results, plots ROC, plots Confusion Matrix using a
    provided threshold, and computes metrics using the provided threshold.
    (Formerly plot_and_evaluate_with_threshold).

    Args:
        data_path_pos (str): Path to positive class results CSV (must contain 'pearson').
        data_path_neg (str): Path to negative class results CSV (must contain 'pearson').
        threshold (float): The pre-calculated classification threshold to use.
        save_path_plots_metrics (str): Directory to save plots.

    Returns:
        dict or None: Dictionary containing metrics ('AUC', 'Threshold_Used',
                      'Accuracy', 'Precision', etc.) or None if evaluation fails.
    """
    print(f"  Attempting evaluation using {data_path_pos} and {data_path_neg}")
    print(f"  Using pre-calculated threshold: {threshold:.6f}")
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

    if 'pearson' not in data.columns:
        print("    ERROR: Evaluation failed. Required 'pearson' column not found in loaded data.")
        print(f"    Columns found: {data.columns.tolist()}")
        return None

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

    # --- ROC Curve Calculation & Plot ---
    print("    Calculating and plotting ROC curve...")
    labels = data['label'].values
    scores = data['pearson'].values
    try:
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
    except Exception as e:
         print(f"    Error calculating ROC curve/AUC: {e}")
         return None

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:0.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right"); plt.grid(alpha=0.5)
    roc_save_path = os.path.join(save_path_plots_metrics, 'roc_curve.png')
    try:
        plt.savefig(roc_save_path)
        print(f"    ROC curve saved to {roc_save_path}")
    except Exception as e: print(f"    Error saving ROC curve plot: {e}")
    plt.close()

    # --- Confusion Matrix Calculation & Plot (using function from visualization.py) ---
    print(f"    Calculating confusion matrix using threshold = {threshold:.6f}...")
    cm_save_path = os.path.join(save_path_plots_metrics, f'confusion_matrix_thresh_{threshold:.4f}.png')
    try:
        # Calls compute_confusion_matrix from visualization.py
        compute_confusion_matrix(data, threshold, cm_save_path)
    except Exception as e: print(f"    Error computing/saving confusion matrix: {e}")

    # --- Metrics Calculation (using function from utils.py) ---
    print(f"    Calculating metrics using threshold = {threshold:.6f}...")
    try:
        # Calls compute_metrics from utils.py
        metrics = compute_metrics(data, threshold)
        print(f"    Metrics calculated: {metrics}")
    except Exception as e:
         print(f"    Error computing metrics: {e}")
         return {'AUC': roc_auc, 'Threshold_Used': threshold} # Return partial results

    metrics['AUC'] = roc_auc
    metrics['Threshold_Used'] = threshold

    return metrics