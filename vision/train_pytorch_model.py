import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


# ============================
#      PARAMÈTRES GLOBAUX
# ============================
DATASET_DIR = "dataset"
BATCH_SIZE = 32
IMG_SIZE = 128
LR = 1e-4
EPOCHS = 30
EARLY_STOP_PATIENCE = 3
MODEL_SAVE_PATH = "model_chess.pth"


# ============================
#     TRANSFORMATIONS
# ============================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# ============================
#         DATASET
# ============================
print("Chargement du dataset…")
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transforms)

# Classes détectées
print(f"Classes détectées : {full_dataset.classes}")

# Split train / validation
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Validation dataset doit utiliser les bons transforms
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)


# ============================
#         MODÈLE
# ============================
model = models.resnet18(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ============================
#      FONCTION TRAINING
# ============================
def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=EPOCHS, patience=EARLY_STOP_PATIENCE):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # ---- EARLY STOPPING ----
        if val_acc > best_acc:
            print(">>> Nouveau meilleur modèle !")
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Aucune amélioration ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("\n⛔ Early stopping déclenché !")
            break

    print("\nEntraînement terminé.")
    model.load_state_dict(best_model_wts)
    return model


# ============================
#     LANCEMENT TRAINING
# ============================
model = train_model(model, train_loader, val_loader, criterion, optimizer)

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nModèle sauvegardé sous : {MODEL_SAVE_PATH}")