import matplotlib.pyplot as plt

def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(predictions, targets, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(predictions, label='Predictions', alpha=0.7)
    plt.plot(targets, label='Ground Truth', alpha=0.7)
    plt.title('Predictions vs Ground Truth')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()
