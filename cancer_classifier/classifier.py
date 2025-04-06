import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
import numpy as np


# ------------------------------
# 1. Define the Dataset
# ------------------------------
class MelanomaDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform
        # Create binary label: 1 if diagnosis contains "melanoma" (case-insensitive), else 0
        self.df['label'] = self.df['Primary Diagnosis'].apply(lambda x: 1 if 'melanoma' in x.lower() else 0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['File Name'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = row['label']
        return image, label


# ------------------------------
# 2. Define Data Augmentation / Transforms
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Create the dataset (assumes CSV is './data.csv' and images are in './scaled_down/')
dataset = MelanomaDataset(csv_file="./ImagePatientType.csv", images_dir="./scaled_down", transform=transform)

# ------------------------------
# 3. Set Up K-Fold Cross Validation
# ------------------------------
k_folds = 10
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
batch_size = 4  # Adjust based on your GPU memory

fold_results = {}

# ------------------------------
# 4. K-Fold Training Loop
# ------------------------------
for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    print(f"\nFOLD {fold}")
    print("--------------------------------")
    # Create subset samplers for train and validation
    train_subset = Subset(dataset, train_ids)
    val_subset = Subset(dataset, val_ids)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # ------------------------------
    # 5. Build the Model using Transfer Learning
    # ------------------------------
    model = models.resnet50(pretrained=True)
    # Freeze all layers in the base model
    for param in model.parameters():
        param.requires_grad = False
    # Replace the final fully connected layer for binary classification (2 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    # ------------------------------
    # 6. Training Loop for this Fold
    # ------------------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_subset)
        print(f"Fold {fold} Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    # ------------------------------
    # 7. Validation Loop for this Fold
    # ------------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    fold_results[fold] = accuracy

# ------------------------------
# 8. Report Overall Results
# ------------------------------
print("\nK-Fold Cross Validation results:")
for fold, acc in fold_results.items():
    print(f"Fold {fold}: {acc:.4f}")
print(f"Average Accuracy: {np.mean(list(fold_results.values())):.4f}")

# ------------------------------
# 9. Example Prediction on a Single Image
# ------------------------------
# (Assuming you want to see a prediction after training on one fold.)
# Here, we load one image from the dataset, preprocess it, and predict its class.
example_img, _ = dataset[0]
example_img = example_img.unsqueeze(0).to(device)  # add batch dimension
model.eval()
with torch.no_grad():
    output = model(example_img)
    _, pred = torch.max(output, 1)
    prediction = "Melanoma" if pred.item() == 1 else "Non-Melanoma"
print(f"Example prediction: {prediction}")
