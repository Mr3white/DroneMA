import torch
import pandas as pd
from utils import normalize

def test_model(model, loader, device, save_path='results.csv'):
    model.eval()
    results = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 应用正则化
            inputs_normalized = normalize(inputs)
            targets_normalized = normalize(targets)
            
            outputs = model(inputs_normalized)
            
            # 收集结果
            batch_results = {
                'inputs': inputs[:, -1, :].cpu().numpy(),
                'targets': targets[:, -1, :].cpu().numpy(),
                'outputs': outputs[:, -1, :].cpu().numpy()
            }
            results.append(batch_results)
    
    # 保存结果
    pd.DataFrame(results).to_csv(save_path)
    print(f"Test results saved to {save_path}")
