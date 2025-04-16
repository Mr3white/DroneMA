import random
from config import config
from data_processing import create_dataloader
from model import R2DGRU
from utils import seed_everything, get_files, create_dirs
from train import train_model

if __name__ == "__main__":
    # 初始化设置
    seed_everything(config.seed)
    create_dirs(config.model_save_path, config.results_save_path)
    
    # 获取数据
    all_positive_files = get_files('../data/positive')

    random.shuffle(all_positive_files)

    split_index = len(all_positive_files) // 2

    train_files = all_positive_files[:split_index] # First half for training
    val_files = all_positive_files[split_index:]

    print(f"Total positive files: {len(all_positive_files)}")
    print(f"Number of training files: {len(train_files)}")
    print(f"Number of validation files: {len(val_files)}")

    train_loader = create_dataloader(train_files, config.train_window, config.batch_size) 
    val_loader = create_dataloader(val_files, config.train_window, config.batch_size)
    
    # 初始化模型
    model = R2DGRU(input_size=1, 
                      hidden_size=32, 
                      output_size=1, 
                      model_type=config.model_type)
    
    # 训练模型
    trained_model = train_model(model, train_loader, val_loader, config) 

    
