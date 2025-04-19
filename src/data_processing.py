# data_processing.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os # Added for file existence check

class RSSIDataset(Dataset):
    """ Custom Dataset for RSSI sequences. """
    def __init__(self, processed_data_list):
        # Expects a list of tuples, where each tuple is (rssi_array, dist_array)
        self.processed_data = processed_data_list

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        # Gets the tuple of numpy arrays for the index
        rssi_np, dist_np = self.processed_data[idx]
        # Convert numpy arrays to tensors here
        # Ensure they are float32 as commonly expected by models
        rssi_tensor = torch.tensor(rssi_np, dtype=torch.float32)
        dist_tensor = torch.tensor(dist_np, dtype=torch.float32)
        # Reshape to (sequence_length, 1) as often needed for RNNs/Transformers
        return rssi_tensor.view(-1, 1), dist_tensor.view(-1, 1)

def preprocess_data(data_df, window_length):
    """Preprocesses data from a single DataFrame using windowing (no Kalman)."""
    if 'rssi' not in data_df.columns or 'dist' not in data_df.columns:
        # Or handle cases where only 'rssi' might be needed if dist is not always present
        raise ValueError("Input DataFrame must contain 'rssi' and 'dist' columns.")

    processed_data_tuples = []
    # Check length before iterating
    if len(data_df) < window_length:
        return [] # Skip files shorter than window length

    # Use .iloc for potentially non-integer index, ensure values are numpy arrays
    rssi_values = data_df['rssi'].values
    dist_values = data_df['dist'].values

    for start_index in range(len(data_df) - window_length + 1):
        end_index = start_index + window_length
        # Slice numpy arrays directly
        window_rssi = rssi_values[start_index:end_index]
        actual_dist = dist_values[start_index:end_index]
        processed_data_tuples.append((window_rssi, actual_dist))
    return processed_data_tuples

def _create_dataloader_base(files, window_length, batch_size, shuffle):
    """Base function for creating dataloaders."""
    all_processed_tuples = []
    print(f"  Preprocessing {len(files)} files for dataloader (shuffle={shuffle})...")
    count = 0
    for file in files:
        count += 1
        # print(f"    Processing file {count}/{len(files)}: {file}") # Verbose logging
        try:
            # Check existence before reading
            if not os.path.exists(file):
                 print(f"    Warning: File not found, skipping: {file}")
                 continue
            data = pd.read_csv(file)
            # Check if data is empty or missing required columns
            if data.empty or not {'rssi', 'dist'}.issubset(data.columns):
                 print(f"    Warning: Skipping file {file} (empty or missing 'rssi'/'dist').")
                 continue

            processed_tuples = preprocess_data(data, window_length)
            all_processed_tuples.extend(processed_tuples)
        except pd.errors.EmptyDataError:
             print(f"    Warning: Skipping empty CSV file: {file}")
        except Exception as e:
            print(f"    Error processing file {file}: {e}")
            # import traceback # Uncomment for detailed debugging
            # traceback.print_exc() # Uncomment for detailed debugging

    if not all_processed_tuples:
         print("  Warning: No data loaded after preprocessing all files. Dataloader will be empty.")
         # Return an empty DataLoader consistent with expected output type
         return DataLoader(RSSIDataset([]), batch_size=batch_size, shuffle=shuffle)

    print(f"  Created {len(all_processed_tuples)} sequences.")
    # *** CORRECTED: Pass the list of tuples directly to the Dataset ***
    dataset = RSSIDataset(all_processed_tuples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Dataloader function for training (shuffle=True)
def create_train_dataloader(files, window_length, batch_size=128):
    """Creates a DataLoader for training (shuffle=True)."""
    # This directly calls the base function with shuffle=True
    return _create_dataloader_base(files, window_length, batch_size, shuffle=True)

# Dataloader function for testing/validation (shuffle=False)
def create_eval_dataloader(files, window_length, batch_size=1):
    """Creates a DataLoader for evaluation/testing (shuffle=False)."""
    # This directly calls the base function with shuffle=False
    return _create_dataloader_base(files, window_length, batch_size, shuffle=False)