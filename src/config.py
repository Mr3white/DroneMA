import torch

class Config:
    seed = 10086
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 20
    train_window = 60
    model_type = 'gru'
    batch_size = 128
    learning_rate = 0.001
    model_save_path = './saved_models/'
    results_save_path = './results/'

config = Config()
