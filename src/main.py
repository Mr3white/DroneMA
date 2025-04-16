# main.py
import random
import os
import pandas as pd
import torch

# --- Configuration and Custom Modules ---
from config import config
from data_processing import create_train_dataloader, create_eval_dataloader
from model import R2DGRU
from test import test_model
from train import train_model
# Import general utils
from utils import (seed_everything, get_files, create_dirs)
# Import the main evaluation/plotting function from visualization
from visualization import plot_and_evaluate # <<< CHANGED IMPORT

# ==================== Main Execution ====================
if __name__ == "__main__":
    # --- 1. Initial Setup ---
    seed_everything(config.seed)
    create_dirs(config.model_save_path, config.results_save_path)
    evaluation_base_path = config.results_save_path
    print(f"Results will be saved under: {evaluation_base_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Get Data & Split for Training/Validation ---
    try:
        all_positive_files = get_files('../data/positive')
        all_negative_files = get_files('../data/negative')
    except FileNotFoundError as e:
        print(f"Error getting data files: {e}")
        exit()
    random.shuffle(all_positive_files)
    split_index = len(all_positive_files) // 2
    train_files = all_positive_files[split_index:]
    val_files = all_positive_files[:split_index]
    print(f"Total positive files: {len(all_positive_files)}")
    print(f"  Training set size (positive): {len(train_files)}")
    print(f"  Validation set size (positive): {len(val_files)}")
    print(f"Total negative files: {len(all_negative_files)}")

    # --- 3. Create DataLoaders for Training ---
    print("\nCreating Training DataLoader...")
    train_loader = create_train_dataloader(train_files, config.train_window, config.batch_size)
    print("Creating Validation DataLoader...")
    val_loader = create_eval_dataloader(val_files, config.train_window, config.batch_size) # Use eval loader (shuffle=False) for validation

    # --- 4. Initialize and Train Model ---
    model = R2DGRU(input_size=1, hidden_size=32, output_size=1, model_type=config.model_type).to(device)

    # *** Define the loss function needed by test_model ***
    loss_function = torch.nn.MSELoss() # Or your actual loss

    print("\nStarting model training...")
    trained_model = train_model(model, train_loader, val_loader, config) 
    print("Model training finished.")

    # --- 5. Evaluation Phase ---
    print("\nStarting evaluation phase...")
    test_window_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    all_final_metrics = []
    print("Generating test result CSV files for each window size...")
    trained_model.to(device).eval()
    test_batch_size = config.batch_size # Or set specifically

    # Get ew_alpha from config (assuming it exists)
    ew_alpha_value = getattr(config, 'ew_alpha', 0) # Default to 0 if not in config

    for test_window in test_window_list:
        print(f"\n  Processing test_window = {test_window}...")
        save_folder = os.path.join(evaluation_base_path, f'test_{test_window}')
        os.makedirs(save_folder, exist_ok=True)

        # --- Generate pos.csv ---
        print(f"    Preparing Positive Validation DataLoader (shuffle=False)...")
        try:
            pos_dataloader = create_eval_dataloader(val_files, test_window, test_batch_size)
            if len(pos_dataloader.dataset) > 0:
                 pos_save_path = os.path.join(save_folder, 'pos.csv')
                 # *** Pass loss_function and ew_alpha to test_model ***
                 test_model(trained_model, pos_dataloader, device, loss_function, save_path=pos_save_path, ew_alpha=ew_alpha_value)
            else:
                 print(f"      Skipping pos.csv generation for window {test_window}: No data after preprocessing validation files.")
        except Exception as e:
            print(f"      Error during pos.csv Dataloader creation or test_model execution: {e}")
            import traceback; traceback.print_exc()

        # --- Generate neg.csv ---
        print(f"    Preparing Negative DataLoader (shuffle=False)...")
        try:
            neg_dataloader = create_eval_dataloader(all_negative_files, test_window, test_batch_size)
            if len(neg_dataloader.dataset) > 0:
                 neg_save_path = os.path.join(save_folder, 'neg.csv')
                 # *** Pass loss_function and ew_alpha to test_model ***
                 test_model(trained_model, neg_dataloader, device, loss_function, save_path=neg_save_path, ew_alpha=ew_alpha_value)
            else:
                 print(f"      Skipping neg.csv generation for window {test_window}: No data after preprocessing negative files.")
        except Exception as e:
            print(f"      Error during neg.csv Dataloader creation or test_model execution: {e}")
            import traceback; traceback.print_exc()

    print("\nFinished generating test result CSV files.")

    # --- 5b. Calculate Metrics using generated CSVs ---
    # The WARNING about incompatibility is now less critical, as test_model generates a 'pearson' column.
    # However, the meaning (batch-level correlation) should be kept in mind when interpreting results.
    print("\nCalculating final metrics using generated pos.csv and neg.csv...")
    metrics_save_path = os.path.join(evaluation_base_path, 'metrics_comparison_IQR.csv')

    for test_window in test_window_list:
        print(f"\n  Attempting evaluation for test_window = {test_window}...")
        data_path_dir = os.path.join(evaluation_base_path, f'test_{test_window}')
        data_path_pos = os.path.join(data_path_dir, 'pos.csv')
        data_path_neg = os.path.join(data_path_dir, 'neg.csv')

        if os.path.exists(data_path_pos) and os.path.exists(data_path_neg):
            # Call the function imported from visualization.py
            metrics = plot_and_evaluate(data_path_pos, data_path_neg, data_path_dir)
            if metrics:
                metrics['Test Window'] = test_window
                all_final_metrics.append(metrics)
                print(f"    Evaluation completed for window {test_window}.") # Removed potentially confusing partial completion message
            else:
                print(f"    Evaluation failed or returned no metrics for window {test_window}. See errors above.")
        else:
            print(f"    Skipping evaluation for window {test_window}: Missing {os.path.basename(data_path_pos)} or {os.path.basename(data_path_neg)}.")

    # --- 5c. Save Aggregated Metrics ---
    if all_final_metrics:
        metrics_df = pd.DataFrame(all_final_metrics)
        # Ensure correct column order, using the actual column names from metrics dict
        cols_order = ['Test Window', 'Threshold_IQR', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity']
        existing_cols = [col for col in cols_order if col in metrics_df.columns]
        extra_cols = [col for col in metrics_df.columns if col not in existing_cols]
        metrics_df = metrics_df[existing_cols + extra_cols]
        metrics_df.to_csv(metrics_save_path, index=False, float_format='%.6f') # Use consistent float format
        print(f"\nAggregated metrics saved to {metrics_save_path}")
    else:
        print("\nNo metrics were calculated successfully across all windows. No summary CSV file generated.")

    print("\nScript finished.")
