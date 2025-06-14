import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch


def plot_loss(train_losses):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracies(train_accs, val_accs, test_accs, early_stop_epoch=None):
    plt.figure(figsize=(8,5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    if early_stop_epoch:
        plt.axvline(x=early_stop_epoch-1, color='r', linestyle='--', label='Early Stop')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def get_node_embeddings(model, data, device):
    model.eval()
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    with torch.no_grad():
        emb = model.conv1(x, edge_index).cpu().numpy()  
    return emb


def plot_node_embeddings(embeddings, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=15,
        alpha=0.8
    )
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.grid(True)
    plt.show()