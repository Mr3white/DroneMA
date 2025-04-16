# 因为只需要关注其趋势，所以对特征和目标值进行正则化再计算的奇思妙想！
import pandas as pd
import numpy as np
import glob
import os
from scipy.ndimage import gaussian_filter1d
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import seed_everything, getFiles
from filterpy.kalman import KalmanFilter
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import pandas as pd
import random

def apply_exponential_weights(series, alpha):
    n = len(series)
    weights = np.exp(alpha * np.arange(n))
    #print(weights)
    normalized_weights = weights / weights.sum()
    #print(normalized_weights)
    weighted_series = series * normalized_weights
    return weighted_series

def split_data(processed_data, test_size=0.2):
    # 将数据索引随机打乱
    total_size = len(processed_data)
    shuffled_indices = np.random.permutation(total_size)
    
    # 计算测试数据的数量
    test_set_size = int(total_size * test_size)
    
    # 划分测试集和训练集
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    # 根据索引抽取对应的数据
    train_data = [processed_data[i] for i in train_indices]
    test_data = [processed_data[i] for i in test_indices]
    
    return train_data, test_data

def normalize(tensor):
    """
    正则化 targets 张量。
    参数:
        targets (torch.Tensor): 需要正则化的 targets 张量，形状应为 (batch_size, window_size, feature_dim)。
    返回:
        torch.Tensor: 正则化后的 targets 张量。
    """
    # 计算每个样本在 window_size 维度上的均值和标准差
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True)
    
    # 避免除以零，添加一个小的epsilon
    #eps = 1e-8
    eps = 0
    tensor_normalized = (tensor - mean) / (std + eps)
    
    return tensor_normalized

def normalize1(tensor):
    """
    正则化 targets 张量到0-1范围内。
    参数:
        tensor (torch.Tensor): 需要正则化的 targets 张量，形状应为 (batch_size, window_size, feature_dim)。
    返回:
        torch.Tensor: 正则化后的 targets 张量。
    """
    # 计算每个样本在 window_size 维度上的最小值和最大值
    min_val = tensor.min(dim=1, keepdim=True)[0]
    max_val = tensor.max(dim=1, keepdim=True)[0]
    
    # 进行最小-最大缩放
    tensor_normalized = (tensor - min_val) / (max_val - min_val + 1e-8)  # 加入一个小的epsilon以避免除以零

    return tensor_normalized

def plot_losses(epoch_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='Epoch Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_losses1(train_losses, test_losses):
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()


def test_plot(dataloader, save_path='test_losses.png'):
    # 存储预测值和真实值
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # 在评估模式下不计算梯度
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            # 正则化 targets
            targets_normalized = normalize(targets)

            outputs = model(inputs)            
            all_predictions.extend(outputs[:,-1,:].view(-1).tolist())
            all_targets.extend(targets_normalized[:,-1,:].view(-1).tolist())

    # 转换为numpy数组方便计算和可视化
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # 计算评估指标
    mse = np.mean((predictions - targets) ** 2)
    print(f'Mean Squared Error: {mse}')

    # 可视化预测值与真实值
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label='Predictions', color='blue')
    plt.plot(targets, label='Actual', color='red', alpha=0.7)
    plt.title('Predictions vs Actual')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.legend()

    # 保存图像到指定路径
    plt.savefig(save_path)
    plt.close()  # 关闭图形，释放内存

def test(dataloader, model, loss_function, save_path='test_losses.csv', ew_alpha=0):
    model.eval()  # 设置模型为评估模式
    model.to(DEVICE)  # 确保模型在GPU上
    all_losses = []  # 存储所有损失

    with torch.no_grad():  # 测试模式，不计算梯度
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            # 正则化 targets
            targets_normalized = normalize(targets)
            inputs_normalized = normalize(inputs)
            #targets_normalized = targets
            outputs = model(inputs)
            loss = loss_function(outputs[:,-1,:], targets_normalized[:,-1,:]).item()  # 计算损失
            predicted = outputs.cpu().numpy().flatten()
            real = targets_normalized.cpu().numpy().flatten()
            if ew_alpha != 0:
                predicted = pd.Series(predicted)
                real = pd.Series(real)
                alpha = ew_alpha
                # 计算应用指数权重后的时间序列
                predicted_weighted = apply_exponential_weights(predicted, alpha)
                real_weighted = apply_exponential_weights(real, alpha)
                # 计算应用权重后的相关系数
                correlation = predicted_weighted.corr(real_weighted)
            else:
                pearson_correlation = pearsonr(predicted, real).statistic
                spearman_correlation = spearmanr(predicted, real).statistic
            all_losses.append((loss, pearson_correlation, spearman_correlation, outputs[:,-1,:].item(), inputs_normalized[:,-1,:].item(), targets_normalized[:,-1,:].item(), inputs[:,-1,:].item(), targets[:,-1,:].item()))
    pd.DataFrame(all_losses, columns=['Loss', 'pearson', 'spearman', 'predicted', 'input_norm' ,'real','input','taget']).to_csv(save_path, index_label='Window Index')
    print(f"Losses saved to {save_path}")

def test1(model, dataloader, loss_function, device):
    model.eval()  # 设置模型为评估模式
    total_test_loss = 0
    with torch.no_grad():  # 在评估模式下，不更新权重
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets_normalized = normalize(targets)
            #targets_normalized = targets
            outputs = model(inputs)
            loss = loss_function(outputs[:, -1:, :], targets_normalized[:, -1:, :])
            total_test_loss += loss.item()
    average_test_loss = total_test_loss / len(dataloader)
    return average_test_loss

seed_everything(42)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_kalman_filter(rssi_values, R=1, Q=1):
    # 创建一个卡尔曼滤波器实例
    kf = KalmanFilter(dim_x=1, dim_z=1)
    
    # 初始化卡尔曼滤波器的参数
    kf.x = np.array([rssi_values.iloc[0]])  # 初始状态 (第一个RSSI值作为初始估计)
    kf.F = np.array([[1]])             # 状态转移矩阵
    kf.H = np.array([[1]])             # 测量函数矩阵
    kf.P *= 1000                       # 协方差矩阵
    kf.R = R                           # 测量噪声
    kf.Q = Q                           # 过程噪声

    # 用于存储滤波后的RSSI值
    filtered_rssi = []

    # 对RSSI序列应用卡尔曼滤波
    for rssi in rssi_values:
        kf.predict()                  # 预测下一个状态
        kf.update(rssi)               # 更新，将新的测量值纳入考虑
        filtered_rssi.append(kf.x[0]) # 保存滤波后的值

    return filtered_rssi

# 预处理函数
def preprocess_data(data, window_length):
    # 应用卡尔曼滤波到RSSI列
    rssi_data = data['rssi']
    filtered_rssi_values = apply_kalman_filter(rssi_data,1,1)
    # 将滤波后的RSSI值添加到原始数据表中
    data['filtered_rssi'] = filtered_rssi_values
    processed_data = []
    for start_index in range(len(data) - window_length + 1):
        end_index = start_index + window_length
        window_rssi = data['rssi'][start_index:end_index].values
        #window_rssi = data['filtered_rssi'][start_index:end_index].values
        #window_remrssi = data['remrssi'][start_index:end_index].values
        #window_rssi = gaussian_filter1d(window_rssi, sigma=1)
        #window_remrssi = gaussian_filter1d(window_remrssi, sigma=1)
        actual_dist = data['dist'][start_index:end_index].values
        processed_data.append((window_rssi,  actual_dist))
    return processed_data


def noLabel_files2Tensor(file_list, sequence_length):
    X0 = np.array([])  # 初始化空的numpy数组

    for file in file_list:
        data = pd.read_csv(file)
        X = preprocess_data(data, sequence_length)
        X = np.array(X)
        if X0.size == 0:
            X0 = X  # 如果X0是空的，直接赋值
        else:
            X0 = np.concatenate((X0, X), axis=0)  # 否则，连接新数据

    if X0.size == 0:
        print("Can't read datasets!")
        return None  # 如果没有数据，返回None

    X_tensor = torch.tensor(X0, dtype=torch.float32)  # 转换为张量
    return X_tensor

# 步骤6: 训练模型
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

num_epochs = 100  # 训练轮次
train_window = 70
#test_window = 60
ew_alpha = 0

model_type = 'gru'

#save_folder = './final/'+str(num_epochs)+'/train_'+str(train_window)+'_test_'+str(test_window)+'_'+str(model_type)+'/'

#save_folder = './final'+'/train_'+str(train_window)+'/'+str(model_type)+'/'+str(num_epochs)+'/'+'test_'+str(test_window)+'/'

positive_files_list = getFiles(folder_path='../data/final/positive',file_pattern='*_position.csv')
negtive_files_list = getFiles(folder_path='../data/final/negtive',file_pattern='*_position.csv')
#train_files_list = getFiles(folder_path='../data/final/positive/train',file_pattern='*_position.csv')
#test_files_list = getFiles(folder_path='../data/final/positive/test',file_pattern='*_position.csv')
#print(positive_files_list)

num_to_select = int(len(positive_files_list) * 0)
test_list = random.sample(positive_files_list, num_to_select)
train_list = list(set(positive_files_list) - set(test_list))

train_data = noLabel_files2Tensor(train_list, sequence_length=train_window)
test_data = noLabel_files2Tensor(test_list, sequence_length=train_window)

# 步骤2: 定义PyTorch Dataset
class RSSIDataset(Dataset):
    def __init__(self, processed_data):
        self.processed_data = processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        rssi, dist = self.processed_data[idx]
        rssi_tensor = rssi.clone().detach().view(-1, 1)
        dist_tensor = dist.clone().detach().view(-1, 1)
        return rssi_tensor, dist_tensor

# 创建训练数据集和测试数据集
train_dataset = RSSIDataset(train_data)
test_dataset = RSSIDataset(test_data)

# 创建 DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class PreprocessingLayer(nn.Module):
    def __init__(self):
        super(PreprocessingLayer, self).__init__()
    
    def forward(self, x):
        # 实现预处理操作：y = y / 1.9 - 127
        return x / 1.9 - 127
    
class NormalizeOverWindow(nn.Module):
    def __init__(self, eps=1e-8):
        super(NormalizeOverWindow, self).__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + self.eps)

class NormalizeOverWindow1(nn.Module):
    def __init__(self, eps=1e-8):
        super(NormalizeOverWindow, self).__init__()
        self.eps = eps

    def forward(self, x):
        min_val = x.min(dim=1, keepdim=True)[0]
        max_val = x.max(dim=1, keepdim=True)[0]
        return (x - min_val) / (max_val - min_val + self.eps)

class ExponentialLayer(nn.Module):
    def __init__(self):
        super(ExponentialLayer, self).__init__()

    def forward(self, x):
        return torch.exp(x * torch.log(torch.tensor(10.0)))

# 步骤4: 定义编码器-解码器模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layer='rnn'):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.preprocess = PreprocessingLayer()
        self.norm = NormalizeOverWindow()
        if hidden_layer == 'rnn':
            self.hidden_layer = nn.RNN(input_size, hidden_size, batch_first=True)
        elif hidden_layer == 'gru':
            self.hidden_layer = nn.GRU(input_size, hidden_size, batch_first=True)
        elif hidden_layer == 'lstm':
            self.hidden_layer = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        #self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.exp = ExponentialLayer()
        #self.net = nn.Sequential(self.fc1, self.sigmoid, self.fc2, self.sigmoid, self.fc3, self.fc4)
        #self.net = nn.Sequential( self.exp, self.fc1, self.fc3)
        #self.net = nn.Sequential( self.relu, self.fc1, self.exp, self.fc3)
        self.net = nn.Sequential(self.fc1, self.exp, self.fc3)
        #self.net = nn.Sequential(self.fc1, self.fc3)

    def forward(self, x):
        #x = self.preprocess(x)
        x = self.norm(x)
        out, _ = self.hidden_layer(x)
        predictions = self.net(out)
        return predictions

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, outputs, targets):
        # 计算余弦相似度
        similarity = self.cosine_similarity(outputs, targets)
        # 将相似度转换为损失
        loss = 1 - similarity
        return loss.mean()  # 返回批次的平均损失

# 步骤5: 初始化模型、损失函数和优化器
model = SimpleLSTM(input_size=1, hidden_size=32, output_size=1, hidden_layer=model_type).to(DEVICE)
loss_function = nn.MSELoss()  # 使用整个序列的MSE损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize a list to store the loss of each epoch
epoch_losses = []
test_losses = []

for epoch in range(num_epochs):
    # 使用tqdm包装数据加载器，以在训练时显示进度条
    loop = tqdm(train_dataloader, leave=True)
    total_epoch_loss = 0
    for inputs, targets in loop:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)  # 确保数据和目标在正确的设备上
        optimizer.zero_grad()
        #inputs = inputs.to(DEVICE)
        #print(targets.shape)
        targets_normalized = normalize(targets)
        #targets_normalized = targets
        outputs = model(inputs)
        #loss = loss_function(outputs[:,-10,:], targets[:,-10,:])
        #print(outputs[:, -1:, :].shape)
        loss = loss_function(outputs[:, -1:, :], targets_normalized[:, -1:, :])  # 使用正则化后的targets
        loss.backward()
        optimizer.step()

        # Accumulate the loss
        total_epoch_loss += loss.item()

        # 更新进度条的描述
        loop.set_description(f'Epoch {epoch+1}/{num_epochs}')
        loop.set_postfix(loss=loss.item())
    # Average the loss over all batches and store it
    average_epoch_loss = total_epoch_loss / len(loop)
    epoch_losses.append(average_epoch_loss)

    # 在每个周期结束后评估测试集
    #average_test_loss = test1( model, test_dataloader, loss_function, device=DEVICE)
    #test_losses.append(average_test_loss)
#plot_losses1(epoch_losses, test_losses)

model_save_folder = './final_f'+'/train_'+str(train_window)+'/'+str(model_type)+'/'+str(num_epochs)+'/'
os.makedirs(model_save_folder, exist_ok=True)
torch.save(model.state_dict(), model_save_folder+'exp.pth')

test_window_list=[10,20,30,40,50,60,70,80,90,100]
for test_window in test_window_list:
    save_folder = model_save_folder+'test_'+str(test_window)+'/'
    # 使用os.makedirs递归创建文件夹
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_folder+"pos/", exist_ok=True)
    os.makedirs(save_folder+"neg/", exist_ok=True)
    processed_pos_data = noLabel_files2Tensor(positive_files_list, sequence_length=test_window)
    pos_dataset = RSSIDataset(processed_pos_data)
    dataloader = DataLoader(pos_dataset, batch_size=1, shuffle=False)
    test(dataloader, model, loss_function, save_path=save_folder+'pos.csv', ew_alpha=ew_alpha)

    processed_neg_data = noLabel_files2Tensor(negtive_files_list, sequence_length=test_window)
    neg_dataset = RSSIDataset(processed_neg_data)
    dataloader = DataLoader(neg_dataset, batch_size=1, shuffle=False)
    test(dataloader, model, loss_function, save_path=save_folder+'neg.csv', ew_alpha=ew_alpha)
'''
    for file in positive_files_list:
        data = noLabel_files2Tensor([file], sequence_length=test_window)
        dataset = RSSIDataset(data)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        test_plot(dataloader, save_path=save_folder+'pos/'+file[-32:-4])
        test(dataloader, model, loss_function, save_path=save_folder+'pos/'+file[-32:-4]+'_result.csv', ew_alpha=ew_alpha)

    for file in negtive_files_list:
        data = noLabel_files2Tensor([file], sequence_length=test_window)
        dataset = RSSIDataset(data)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        test_plot(dataloader, save_path=save_folder+'neg/'+file[-32:-4])
        test(dataloader, model, loss_function, save_path=save_folder+'neg/'+file[-32:-4]+'_result.csv', ew_alpha=ew_alpha)
'''