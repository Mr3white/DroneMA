# main.py
import random
import os
import pandas as pd
import torch

# --- Configuration and Custom Modules ---
from config import config
from data_processing import create_train_dataloader, create_eval_dataloader
from model import R2DGRU
# Import test_model AND the new evaluate function from test.py
from test import test_model, evaluate # <<< MODIFIED IMPORT
from train import train_model
# Import utils needed here
from utils import (seed_everything, get_files, create_dirs,
                   calculate_thresholds_from_train_data) # evaluate no longer imported from here

# ==================== Main Execution ====================
if __name__ == "__main__":
    # --- 1. Initial Setup ---
    seed_everything(config.seed)
    create_dirs(config.model_save_path, config.results_save_path)
    evaluation_base_path = config.results_save_path
    temp_threshold_path = os.path.join(evaluation_base_path, "temp_train_eval_for_threshold")
    print(f"Results will be saved under: {evaluation_base_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Get Data & Split ---
    try:
        all_positive_files = get_files('../data/positive')
        all_negative_files = get_files('../data/negative')
    except FileNotFoundError as e: print(f"Error getting data files: {e}"); exit()
    random.shuffle(all_positive_files)
    split_index = len(all_positive_files) // 2
    train_files = all_positive_files[:split_index]
    val_files = all_positive_files[split_index:]
    print(f"Total positive files: {len(all_positive_files)}")
    print(f"  Training set size (positive): {len(train_files)}")
    print(f"  Validation set size (positive): {len(val_files)}")
    print(f"Total negative files: {len(all_negative_files)}")

    # --- 3. Create DataLoaders for Training ---
    print("\nCreating Training DataLoader...")
    train_loader = create_train_dataloader(train_files, config.train_window, config.batch_size)
    print("Creating Validation DataLoader...")
    val_loader = create_eval_dataloader(val_files, config.train_window, config.batch_size)

    # --- 4. Initialize and Train Model ---
    model = R2DGRU(input_size=1, hidden_size=32, output_size=1, model_type=config.model_type).to(device)
    loss_function = torch.nn.MSELoss()
    print("\nStarting model training...")
    trained_model = train_model(model, train_loader, val_loader, config)
    print("Model training finished.")

    # --- 5. Calculate Thresholds from Training Data ---
    test_window_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ew_alpha_value = getattr(config, 'ew_alpha', 0)
    threshold_dict = calculate_thresholds_from_train_data(
        train_files=train_files, window_list=test_window_list, model=trained_model,
        create_dataloader_func=create_eval_dataloader, test_func=test_model,
        device=device, loss_fn=loss_function, ew_alpha=ew_alpha_value,
        temp_path=temp_threshold_path
    )
    print("\nCalculated Thresholds per Window:", threshold_dict)

    # --- 6. Final Evaluation on Test/Validation Set ---
    print("\nStarting final evaluation phase using calculated thresholds...")
    all_final_metrics = []
    print("Generating test set result CSV files (pos.csv/neg.csv)...")
    trained_model.to(device).eval()
    test_batch_size = 1

    for test_window in test_window_list:
        print(f"\n  Processing test_window = {test_window} for final evaluation...")
        save_folder = os.path.join(evaluation_base_path, f'test_{test_window}')
        os.makedirs(save_folder, exist_ok=True)
        pos_save_path = os.path.join(save_folder, 'pos.csv')
        neg_save_path = os.path.join(save_folder, 'neg.csv')

        # --- Generate pos.csv ---
        print(f"    Preparing Positive Validation DataLoader (shuffle=False)...")
        try:
            pos_dataloader = create_eval_dataloader(val_files, test_window, test_batch_size)
            if len(pos_dataloader.dataset) > 0: test_model(trained_model, pos_dataloader, device, loss_function, save_path=pos_save_path, ew_alpha=ew_alpha_value)
            else: print(f"      Skipping pos.csv generation: No data.")
        except Exception as e: print(f"      Error generating pos.csv: {e}"); import traceback; traceback.print_exc()

        # --- Generate neg.csv ---
        print(f"    Preparing Negative DataLoader (shuffle=False)...")
        try:
            neg_dataloader = create_eval_dataloader(all_negative_files, test_window, test_batch_size)
            if len(neg_dataloader.dataset) > 0: test_model(trained_model, neg_dataloader, device, loss_function, save_path=neg_save_path, ew_alpha=ew_alpha_value)
            else: print(f"      Skipping neg.csv generation: No data.")
        except Exception as e: print(f"      Error generating neg.csv: {e}"); import traceback; traceback.print_exc()

    print("\nFinished generating test set result CSV files.")

    # --- Calculate Metrics using generated CSVs and PRE-CALCULATED Thresholds ---
    print("\nCalculating final metrics using generated pos.csv/neg.csv and train-derived thresholds...")
    metrics_save_path = os.path.join(evaluation_base_path, 'metrics_comparison_IQR_TrainThreshold.csv')

    for test_window in test_window_list:
        print(f"\n  Attempting evaluation for test_window = {test_window}...")
        data_path_dir = os.path.join(evaluation_base_path, f'test_{test_window}')
        data_path_pos = os.path.join(data_path_dir, 'pos.csv')
        data_path_neg = os.path.join(data_path_dir, 'neg.csv')
        threshold_for_window = threshold_dict.get(test_window)

        if threshold_for_window is None:
             print(f"    Skipping evaluation: Threshold not calculated."); continue

        if os.path.exists(data_path_pos) and os.path.exists(data_path_neg):
            # Call the evaluate function imported from test.py <<< CHANGED CALL
            metrics = evaluate(
                data_path_pos,
                data_path_neg,
                threshold=threshold_for_window,
                save_path_plots_metrics=data_path_dir
            )
            if metrics:
                metrics['Test Window'] = test_window
                all_final_metrics.append(metrics)
                print(f"    Evaluation completed for window {test_window}.")
            else:
                print(f"    Evaluation failed or returned no metrics for window {test_window}.")
        else:
            print(f"    Skipping evaluation: Missing {os.path.basename(data_path_pos)} or {os.path.basename(data_path_neg)}.")

    # --- Save Aggregated Metrics ---
    if all_final_metrics:
        metrics_df = pd.DataFrame(all_final_metrics)
        cols_order = ['Test Window', 'Threshold_Used', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity']
        existing_cols = [col for col in cols_order if col in metrics_df.columns]
        extra_cols = [col for col in metrics_df.columns if col not in existing_cols]
        metrics_df = metrics_df[existing_cols + extra_cols]
        metrics_df.to_csv(metrics_save_path, index=False, float_format='%.6f')
        print(f"\nAggregated metrics saved to {metrics_save_path}")
    else:
        print("\nNo metrics were calculated successfully. No summary CSV file generated.")

    print("\nScript finished.")