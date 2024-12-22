import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import classification_report
from PIL import Image
import os

# Define parameters
data_dir = r"C:\Users\kvidi\Desktop\whack24\ironyman\Palm"  # Replace with your dataset path
csv_file = r"C:\Users\kvidi\Desktop\whack24\ironyman\labels.csv"  # Replace with the path to your CSV file
batch_size = 32
learning_rate = 0.001
num_epochs = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom dataset class to load images and labels from CSV
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)  # Load the CSV with filenames and labels
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = {'Non-Anemic': 0, 'Anemic': 1 , 'Unknown' : 0}  # Ensure proper label mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])  # Get the image filename
        image = Image.open(img_name).convert("RGB")  # Open the image
        label = self.df.iloc[idx, 1]  # Get the label (string)
        
        # Convert the label using the label_map
        label = self.label_map.get(label, -1)  # Default to -1 for unknown labels (you can handle this however you want)
        
        # If the label is -1, it means it's an unexpected label, handle it gracefully
        if label == -1:
            print(f"Warning: Unexpected label '{self.df.iloc[idx, 1]}' at index {idx}.")
        
        if self.transform:
            image = self.transform(image)

        return image, label

# Load dataset
dataset = CustomImageDataset(csv_file=csv_file, root_dir=data_dir, transform=data_transforms)

# Split dataset (7:2:1 rule)
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained ResNet50
model = models.resnet50(pretrained=True)

# Modify the final fully connected layer for binary classification (assuming 2 classes)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Two classes: Anemic and Non-Anemic
model = model.to(device)  # Move the model to the selected device (GPU or CPU)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and validation function
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the selected device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {train_loss:.4f} Acc: {train_accuracy:.4f}")

        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the selected device
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / len(val_loader.dataset)
        val_accuracy = running_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f}")

        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Best Val Accuracy: {best_accuracy:.4f}")

# Test function
def test_model(model, test_loader):
    print("\nTesting the model...")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    running_corrects = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the selected device
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = running_corrects.double() / len(test_loader.dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

# Train, validate, and test the model
train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs)
test_model(model, test_loader)
