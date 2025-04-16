import torch
from tqdm import tqdm
from utils import seed_everything, normalize
from visualization import plot_losses

def train_model(model, train_loader, test_loader, config):
    model.to(config.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}'):

            inputs, targets = inputs.to(config.device), targets.to(config.device)
            
            inputs_normalized = normalize(inputs)
            targets_normalized = normalize(targets)
            
            optimizer.zero_grad()
            outputs = model(inputs_normalized)
            loss = criterion(outputs[:, -1, :], targets_normalized[:, -1, :])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        avg_test_loss = evaluate(model, test_loader, criterion, config.device)
        test_losses.append(avg_test_loss)
        
        print(f'Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
    
    plot_losses(train_losses, test_losses)
    torch.save(model.state_dict(), f'{config.model_save_path}model.pth')
    return model

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            inputs_normalized = normalize(inputs)
            targets_normalized = normalize(targets)
            
            outputs = model(inputs_normalized)
            loss = criterion(outputs[:, -1, :], targets_normalized[:, -1, :])
            total_loss += loss.item()
    
    return total_loss / len(loader)