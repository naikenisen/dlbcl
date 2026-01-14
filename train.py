import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import v2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, PngImagePlugin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Augmenter la limite de decompression PNG pour les gros profils ICC
PngImagePlugin.MAX_TEXT_CHUNK = 10 * (1024**2)  # 10 MB

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 8
learning_rate = 0.0001
num_epochs = 100


class OSDataset(Dataset):
    """Dataset personnalise pour la prediction de l'Overall Survival"""
    
    def __init__(self, image_paths, os_values, transform=None, max_size=512):
        self.image_paths = image_paths
        self.os_values = os_values
        self.transform = transform
        self.max_size = max_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Charger l'image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Redimensionner proportionnellement si trop grande (pour memoire GPU)
        w, h = img.size
        if max(w, h) > self.max_size:
            ratio = self.max_size / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Appliquer les transformations
        if self.transform:
            img = self.transform(img)
            
        # Recuperer la valeur OS
        os_value = torch.tensor(self.os_values[idx], dtype=torch.float32)
        
        return img, os_value


# Fonction de collate pour gerer des tailles d'images variables
def collate_fn(batch):
    """Padding des images pour avoir la même taille dans un batch"""
    images, targets = zip(*batch)
    
    # Trouver les dimensions maximales
    max_h = max([img.shape[1] for img in images])
    max_w = max([img.shape[2] for img in images])
    
    # Padding des images
    padded_images = []
    for img in images:
        c, h, w = img.shape
        padded = torch.zeros((c, max_h, max_w))
        padded[:, :h, :w] = img
        padded_images.append(padded)
    
    return torch.stack(padded_images), torch.stack(targets)


# Charger les donnees cliniques
clinical_data = pd.read_csv('clinical_data.csv')

# Recuperer tous les fichiers images
image_dir = Path('dataset')
image_files = list(image_dir.glob('*.png'))

# Creer un mapping patient_id OS
patient_os_map = dict(zip(clinical_data['patient_id'], clinical_data['OS']))

# Filtrer les images qui ont une valeur OS disponible
valid_images = []
os_values = []

for img_path in image_files:
    # Extraire le patient_id du nom de fichier (ex: "17658.png" ou "17658_2.png")
    patient_id = int(img_path.stem.split('_')[0])
    
    if patient_id in patient_os_map:
        os_val = patient_os_map[patient_id]
        # Verifier que la valeur OS n'est pas NaN
        if pd.notna(os_val):
            valid_images.append(str(img_path))
            os_values.append(float(os_val))

print(f"Nombre d'images avec donnees OS: {len(valid_images)}")
print(f"OS min: {min(os_values):.2f}, OS max: {max(os_values):.2f}, OS moyen: {np.mean(os_values):.2f}")

# Split train/validation
train_imgs, valid_imgs, train_os, valid_os = train_test_split(
    valid_images, os_values, test_size=0.2, random_state=42
)

print(f"Train: {len(train_imgs)} images, Validation: {len(valid_imgs)} images")

# Define transforms (pas de resize fixe)
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(90),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets (max_size=1024 pour garder plus de details)
train_dataset = OSDataset(train_imgs, train_os, transform=train_transforms, max_size=1024)
valid_dataset = OSDataset(valid_imgs, valid_os, transform=valid_transforms, max_size=1024)

# Data loaders avec collate_fn pour gerer les tailles variables
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                          collate_fn=collate_fn, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                          collate_fn=collate_fn, num_workers=2)


# Modele ResNet pour la regression avec support de tailles variables
class ResNetRegression(nn.Module):
    def __init__(self):
        super(ResNetRegression, self).__init__()
        # Charger ResNet50 pre-entraîne (plus profond que ResNet18)
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        
        # Retirer la couche fc et avgpool originales
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Pooling adaptatif pour gerer differentes tailles d'entree
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Nouvelle tête pour la regression
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),  # ResNet50 a 2048 features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Sortie unique pour la regression
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.regressor(x)
        return x.squeeze(1)  # Shape: (batch_size,)


model = ResNetRegression().to(device)
print(f"Modele ResNet50 charge pour regression")

# Loss function et optimizer pour la regression
criterion = nn.MSELoss()  # Mean Squared Error pour la regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

best_mae = float('inf')

# Listes pour tracer les courbes
train_losses = []
valid_maes = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    
    for inputs, targets in train_loader_tqdm:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=running_loss/len(train_loader))
    
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')
    
    # Validation
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        valid_loader_tqdm = tqdm(valid_loader, desc="Validating", unit="batch")
        
        for inputs, targets in valid_loader_tqdm:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calcul des metriques de regression
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    
    valid_maes.append(mae)
    
    print(f'Validation MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')
    
    # Scheduler
    scheduler.step(mae)
    
    # Sauvegarder le meilleur modele base sur MAE
    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), 'best_model_regression.pth')
        print(f'Nouveau meilleur modele sauvegarde (MAE: {mae:.4f})')
        
        # Creer un scatter plot des predictions vs valeurs reelles
        plt.figure(figsize=(10, 6))
        plt.scatter(all_targets, all_preds, alpha=0.5)
        plt.plot([min(all_targets), max(all_targets)], 
                 [min(all_targets), max(all_targets)], 
                 'r--', lw=2, label='Prediction parfaite')
        plt.xlabel('OS reel (annees)')
        plt.ylabel('OS predit (annees)')
        plt.title(f'Predictions vs Valeurs Reelles (MAE: {mae:.4f}, R²: {r2:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('predictions_vs_actual.png')
        plt.close()

# Tracer les courbes d'apprentissage
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(valid_maes, label='Validation MAE', color='orange')
plt.xlabel('Epoch')
plt.ylabel('MAE (annees)')
plt.title('Validation MAE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves_regression.png')
plt.show()

print("Entrainement termine")
print(f"Meilleur MAE: {best_mae:.4f}")
