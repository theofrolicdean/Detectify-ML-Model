import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn
from timm import create_model
import os

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.3, pretrained=True):
        super().__init__()
        self.base_model = create_model("tf_efficientnetv2_l", pretrained=pretrained, num_classes=0)
        num_features = self.base_model.num_features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        self.freeze_layers()

    def freeze_layers(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in list(self.base_model.parameters())[-30:]:
            param.requires_grad = True

    def forward(self, x):
        features = self.base_model.forward_features(x)
        out = self.classifier(features)
        if out.dim() == 2 and out.size(1) == 1:
            out = out.squeeze(1)
        return out


def train_model(model, train_loader, valid_loader, num_epochs=10, 
                device="cpu", resume_checkpoint=None
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    os.makedirs("saved_models", exist_ok=True)

    start_epoch = 0
    best_val_loss = float("inf")
    patience, epochs_without_improve = 5, 0

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    # Resume
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["val_loss"]

        print(f"Resuming from epoch {start_epoch}")

    # Train
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.view(-1)).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # VALIDATION 
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.float())

                val_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (preds == labels.view(-1)).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(val_acc)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
            f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improve = 0
            save_model(epoch, model, optimizer, train_loss, val_loss, device)
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= patience:
            print("Early stopping")
            break

        scheduler.step()

    return train_losses, valid_losses, train_accuracies, valid_accuracies


def save_model(epoch, model, optimizer, train_loss, val_loss, device):
    torch.save(model.state_dict(), f"saved_models/model_epoch_{epoch+1}_weights.pth")
    torch.save(model, f"saved_models/model_epoch_{epoch+1}_full.pth")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,   
        "val_loss": val_loss
    }, f"saved_models/checkpoint_epoch_{epoch + 1}.pth")

    model.eval()
    example_input = torch.randn(1, 3, 224, 224).to(device)
    traced = torch.jit.trace(model, example_input)
    traced.save(f"saved_models/model_epoch_{epoch + 1}_traced.pt")
