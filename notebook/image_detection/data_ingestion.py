import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

class DeepFakeDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label_dir in os.listdir(dataset_dir):
            label_path = os.path.join(dataset_dir, label_dir)
            if os.path.isdir(label_path):
                label = 0 if label_dir == "real" else 1
                for image_name in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_name)
                    if image_path.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.images.append(image_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx])
            if image.mode != "RGB":
                image = image.convert("RGB")

            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {self.images[idx]}: {str(e)}")
            placeholder_image = torch.zeros((3, 224, 224)) if self.transform else Image.new("RGB", (224, 224), color="black")
            return placeholder_image, self.labels[idx]

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, valid_transform

def prepare_data(dataset_dir='data', batch_size=32):
    split_data_dir = os.path.join(dataset_dir, "split_data")

    if not os.path.exists(split_data_dir):
        print("Creating split data directories...")
        os.makedirs(split_data_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            for label in ["real", "fake"]:
                os.makedirs(os.path.join(split_data_dir, split, label), exist_ok=True)

        for label in ["real", "fake"]:
            source_dir = os.path.join(dataset_dir, label)
            images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            train, temp = train_test_split(images, test_size=0.2, random_state=42)
            val, test = train_test_split(temp, test_size=0.5, random_state=42)

            for split, split_imgs in zip(["train", "val", "test"], [train, val, test]):
                for img in tqdm(split_imgs, desc=f"Copying {label} {split} images"):
                    shutil.copy2(os.path.join(source_dir, img), os.path.join(split_data_dir, split, label, img))

    train_tf, valid_tf = get_transforms()
    train_ds = DeepFakeDataset(os.path.join(split_data_dir, "train"), transform=train_tf)
    val_ds = DeepFakeDataset(os.path.join(split_data_dir, "val"), transform=valid_tf)
    test_ds = DeepFakeDataset(os.path.join(split_data_dir, "test"), transform=valid_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
