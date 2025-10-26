# animal_classifier_kaggle.py
import os
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------
# Path to the folder that contains the breed sub-folders
DATA_ROOT = 'C:\\Users\\NIHARIKA\\OneDrive\\Desktop\\VS Code\\lmage based Animal Type Classification for cattle and buffaloes\\dataset\\Indian_bovine_breeds\\Indian_bovine_breeds'

# Which breeds are buffaloes? Add/remove names as needed.
BUFFALO_BREEDS = {
    'Jaffrabadi', 'Murrah', 'Mehsana', 'Banni',  # example buffalo breeds
    # add more if you know the exact names
}

# Transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

BATCH_SIZE = 32
NUM_EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# --------------------------------------------------------------
# 2. CUSTOM DATASET (maps breed folder → binary label)
# --------------------------------------------------------------
class BovineDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []                # (image_path, label)
        self.class_names = ['cow', 'buffalo']

        for breed_folder in self.root_dir.iterdir():
            if not breed_folder.is_dir():
                continue
            breed_name = breed_folder.name
            label = 1 if breed_name in BUFFALO_BREEDS else 0   # 0 = cow, 1 = buffalo

            for img_path in breed_folder.glob('*.png'):       # PNG files
                self.samples.append((str(img_path), label))

        random.shuffle(self.samples)
        print(f'Found {len(self.samples)} PNG images '
              f'({sum(l for _, l in self.samples if l == 0)} cow, '
              f'{sum(l for _, l in self.samples if l == 1)} buffalo)')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# --------------------------------------------------------------
# 3. BUILD DATASET & SPLIT
# --------------------------------------------------------------
full_dataset = BovineDataset(root_dir=DATA_ROOT, transform=val_test_transform)

total = len(full_dataset)
train_size = int(0.70 * total)
val_size   = int(0.15 * total)
test_size  = total - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    full_dataset, [train_size, val_size, test_size]
)

# Apply different transforms to the training portion
train_ds.dataset.transform = train_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

# --------------------------------------------------------------
# 4. MODEL
# --------------------------------------------------------------
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)      # 2 classes
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# --------------------------------------------------------------
# 5. TRAINING LOOP
# --------------------------------------------------------------
def train_model():
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- train ----
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # ---- validation ----
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        epoch_val_loss = running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        scheduler.step()
        print(f'Epoch {epoch:02d} | Train loss: {epoch_train_loss:.4f} | Val loss: {epoch_val_loss:.4f}')

        # save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_animal_classifier.pth')

    # plot losses
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training / Validation Loss')
    plt.show()

# --------------------------------------------------------------
# 6. EVALUATION
# --------------------------------------------------------------
def evaluate():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f'\nTest Accuracy: {acc:.4f}')
    print(classification_report(all_labels, all_preds,
                                target_names=full_dataset.class_names))

# --------------------------------------------------------------
# 7. SINGLE-IMAGE PREDICTION (example)
# --------------------------------------------------------------
def predict_one(image_path):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img_tensor = val_test_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img_tensor)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        pred_idx = out.argmax().item()
    return full_dataset.class_names[pred_idx], probs[pred_idx].item()

# --------------------------------------------------------------
# 8. RUN
# --------------------------------------------------------------
if __name__ == '__main__':
    train_model()
    model.load_state_dict(torch.load('best_animal_classifier.pth'))
    evaluate()

    # Example prediction – pick any PNG file from any breed folder
    example_img = Path(DATA_ROOT) / 'Murrah' / os.listdir(Path(DATA_ROOT) / 'Murrah')[0]
    pred_class, conf = predict_one(str(example_img))
    print(f'\nExample: {example_img.name} → {pred_class} (confidence {conf:.2%})')