from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

EVAL_RESULT_PATH = "results"

def evaluate_model(model, test_loader, device="cuda"):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    test_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            test_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.view(-1)).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    test_loss /= total
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Loss: {test_loss}, Acc: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")

def plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies, eval_path=EVAL_RESULT_PATH):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "b", label="Train Loss")
    plt.plot(epochs, valid_losses, "r", label="Val Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, "b", label="Train Acc")
    plt.plot(epochs, valid_accuracies, "r", label="Val Acc")
    plt.title("Accuracy Over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig(eval_path)
    plt.show()