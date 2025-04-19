# visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings
import torch # Added for type checking in plot_predictions

# --- Sklearn for plotting ---
# Only need confusion matrix related imports now
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Configuration (Optional) ---
# from config import config

# Suppress FutureWarning related to sklearn potentially
warnings.filterwarnings('ignore', category=FutureWarning)

# === Plotting Functions ===

def plot_losses(train_losses, test_losses, save_path=None):
    """Plots training and validation/test losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Validation/Test Loss')
    plt.title('Training and Validation/Test Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    if save_path:
        try: plt.savefig(save_path); print(f"Loss plot saved to {save_path}")
        except Exception as e: print(f"Error saving loss plot to {save_path}: {e}")
    plt.close()

def plot_predictions(predictions, targets, save_path=None):
    """Plots model predictions against ground truth."""
    plt.figure(figsize=(12, 6))
    if isinstance(predictions, torch.Tensor): predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor): targets = targets.detach().cpu().numpy()
    plt.plot(predictions.flatten(), label='Predictions', alpha=0.7)
    plt.plot(targets.flatten(), label='Ground Truth', alpha=0.7)
    plt.title('Predictions vs Ground Truth')
    plt.xlabel('Sample Index / Time Steps'); plt.ylabel('Value')
    plt.legend()
    if save_path:
        try: plt.savefig(save_path); print(f"Prediction plot saved to {save_path}")
        except Exception as e: print(f"Error saving prediction plot to {save_path}: {e}")
    plt.close()


def compute_confusion_matrix(data, threshold, save_path_file):
    """
    Computes, plots, and saves the confusion matrix based on 'pearson' column
    and a given threshold. (Plotting part of the evaluation workflow).
    """
    if 'pearson' not in data.columns or 'label' not in data.columns:
        raise ValueError("Evaluation Error: DataFrame must contain 'pearson' and 'label' columns for compute_confusion_matrix.")
    valid_data = data.dropna(subset=['pearson'])
    if valid_data.empty:
        print("Error: No valid data points remaining for confusion matrix calculation.")
        return
    predictions = (valid_data['pearson'] >= threshold).astype(int)
    labels = valid_data['label']
    if len(np.unique(labels)) < 2 : print(f"Warning: Only one class present in true labels for confusion matrix (Threshold={threshold:.3f}).")
    elif len(np.unique(predictions)) < 2 : print(f"Warning: Only one class predicted for confusion matrix (Threshold={threshold:.3f}).")
    try:
        unique_labels_in_data = sorted(labels.unique())
        cm = confusion_matrix(labels, predictions, labels=unique_labels_in_data)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels_in_data)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix (Threshold={threshold:.3f})')
        plt.savefig(save_path_file)
        plt.close()
        print(f"    Confusion matrix saved to {save_path_file}")
    except Exception as e: print(f"Error calculating or plotting confusion matrix: {e}")

# Removed plot_and_evaluate_with_threshold function from here