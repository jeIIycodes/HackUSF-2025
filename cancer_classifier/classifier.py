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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

# Create the dataset (CSV file and images directory)
dataset = MelanomaDataset(csv_file="./ImagePatientType.csv", images_dir="./scaled_down", transform=transform)

# ------------------------------
# 3. Set Up K-Fold Cross Validation (for classification training)
# ------------------------------
k_folds = 2  # using 2 folds for brevity in this example
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 1  # use a small number for demonstration
batch_size = 4

fold_results = {}

# ------------------------------
# 4. K-Fold Training Loop for Classification
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
    # 5. Build the Model using Transfer Learning (Classification)
    # ------------------------------
    model = models.resnet50(pretrained=True)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace the final fully connected layer for binary classification (2 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    # ------------------------------
    # 6. Training Loop for this Fold (Classification)
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
        print(f"Fold {fold} Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    # ------------------------------
    # 7. Validation Loop for this Fold (Classification)
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
# 8. Report Overall Classification Results
# ------------------------------
print("\nK-Fold Cross Validation results (Classification):")
for fold, acc in fold_results.items():
    print(f"Fold {fold}: {acc:.4f}")
print(f"Average Accuracy: {np.mean(list(fold_results.values())):.4f}")

# ------------------------------
# 9. Example Prediction on a Single Image (Classification)
# ------------------------------
example_img, _ = dataset[0]
example_img = example_img.unsqueeze(0).to(device)  # add batch dimension
model.eval()
with torch.no_grad():
    output = model(example_img)
    _, pred = torch.max(output, 1)
    prediction = "Melanoma" if pred.item() == 1 else "Non-Melanoma"
print(f"Example prediction (Classification): {prediction}")

# ------------------------------
# 10. Train Final Classification Model on All Data
# ------------------------------
print("\nTraining final classification model on all data...")
full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
final_model = models.resnet50(pretrained=True)
for param in final_model.parameters():
    param.requires_grad = False
final_model.fc = nn.Linear(final_model.fc.in_features, 2)
final_model = final_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.fc.parameters(), lr=1e-3)

for epoch in range(15):  # adjust epochs as needed
    final_model.train()
    for images, labels in full_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = final_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save the final classification model
torch.save(final_model.state_dict(), "melanoma_classifier.pth")
print("Final classification model saved as 'melanoma_classifier.pth'")

# ------------------------------
# 11. PCA Visualization for Classification Model
# ------------------------------
print("Generating PCA plot for classification model...")
features = []
colors = []

pca_model = models.resnet50(pretrained=True)
for param in pca_model.parameters():
    param.requires_grad = False
pca_model.fc = nn.Identity()
pca_model = pca_model.to(device)
pca_model.eval()

with torch.no_grad():
    for i in range(len(dataset)):
        image, _ = dataset[i]
        img_tensor = image.unsqueeze(0).to(device)
        feat = pca_model(img_tensor).cpu().numpy().flatten()
        features.append(feat)
        diag = dataset.df.iloc[i]['Primary Diagnosis'].lower()
        if "malignant melanoma" in diag:
            colors.append("red")
        elif "carcinoma" in diag:
            colors.append("blue")
        else:
            colors.append("green")

pca = PCA(n_components=2)
reduced = pca.fit_transform(features)

plt.figure(figsize=(10, 7))
plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, s=100, alpha=0.7, edgecolor='k')
plt.title("2D PCA of Image Features (Classification Model)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_plot.png")
plt.show()

# ------------------------------
# 12. Contrastive Training on Full Data
# ------------------------------
print("\nTraining contrastive model on all data...")

# Define a contrastive model with a projection head
class ContrastiveModel(nn.Module):
    def __init__(self, projection_dim=128):
        super(ContrastiveModel, self).__init__()
        base_model = models.resnet50(pretrained=True)
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.fc = nn.Identity()  # extract features from the base
        self.base = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    def forward(self, x):
        features = self.base(x)
        projection = self.projection_head(features)
        return features, projection

contrastive_model = ContrastiveModel(projection_dim=128).to(device)
optimizer_contrast = optim.Adam(contrastive_model.projection_head.parameters(), lr=1e-3)

# Define the Supervised Contrastive Loss
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, eps=1e-12):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps
    def forward(self, features, labels):
        # Normalize features
        features = nn.functional.normalize(features, dim=1)
        batch_size = features.shape[0]
        logits = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        # Remove self-comparisons from mask
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(features.device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)
        loss = -mean_log_prob_pos.mean()
        return loss

sup_con_loss = SupConLoss(temperature=0.07)
contrastive_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_contrast_epochs = 20  # adjust epochs as needed

for epoch in range(num_contrast_epochs):
    contrastive_model.train()
    running_loss = 0.0
    for images, labels in contrastive_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer_contrast.zero_grad()
        _, projections = contrastive_model(images)
        loss = sup_con_loss(projections, labels)
        loss.backward()
        optimizer_contrast.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Contrastive Epoch {epoch+1}/{num_contrast_epochs} - Loss: {epoch_loss:.4f}")

# Save the contrastive model
torch.save(contrastive_model.state_dict(), "melanoma_contrastive_model.pth")
print("Contrastive model saved as 'melanoma_contrastive_model.pth'")

# ------------------------------
# 13. PCA Visualization with Contrastive Model
# ------------------------------
print("Generating PCA plot with contrastive model...")
features_contrast = []
colors_contrast = []

contrastive_model.eval()
with torch.no_grad():
    for i in range(len(dataset)):
        image, _ = dataset[i]
        img_tensor = image.unsqueeze(0).to(device)
        feat, _ = contrastive_model(img_tensor)
        feat_np = feat.cpu().numpy().flatten()
        features_contrast.append(feat_np)
        diag = dataset.df.iloc[i]['Primary Diagnosis'].lower()
        if "malignant melanoma" in diag:
            colors_contrast.append("red")
        elif "carcinoma" in diag:
            colors_contrast.append("blue")
        else:
            colors_contrast.append("green")

pca_contrast = PCA(n_components=2)
reduced_contrast = pca_contrast.fit_transform(features_contrast)

plt.figure(figsize=(10, 7))
plt.scatter(reduced_contrast[:, 0], reduced_contrast[:, 1], c=colors_contrast, s=100, alpha=0.7, edgecolor='k')
plt.title("2D PCA of Image Features (Contrastive Model)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_contrast_plot.png")
plt.show()
