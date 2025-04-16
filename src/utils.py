import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import glob
import random
import os
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_files(folder_path, extension='csv'):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list

def create_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)
        
def normalize(tensor, eps=1e-8):
    """
    窗口级正则化 (batch_size, seq_len, features) -> 同形状正则化张量
    """
    # 计算每个样本在序列维度上的统计量
    mean = tensor.mean(dim=1, keepdim=True)  # 保持维度用于广播
    std = tensor.std(dim=1, keepdim=True)
    return (tensor - mean) / (std + eps)
