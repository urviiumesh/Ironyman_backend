import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset
class EyesAnemiaDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Extract relevant features
        self.image_names = self.data['IMAGE_ID']
        self.hb_levels = self.data['HB_LEVEL'].values.astype('float32')  # Extract HB_LEVEL
        self.labels = self.data['REMARK'].apply(lambda x: 1 if x == 'Anemic' else 0).values  # Binary classification

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_names[idx] + ".png")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Load HB_LEVEL and anemia label
        hb_level = torch.tensor(self.hb_levels[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, hb_level, label


# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resizing to 224x224 for ResNet
    transforms.ToTensor(),           # Converting image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
])

# Dataset and DataLoader
image_dir = r"C:\Users\kvidi\Desktop\whack24\ironyman\eyes\CP-AnemiC (A Conjunctival Pallor) Dataset from Ghana\CP-AnemiC dataset\combined"  
csv_file = r"C:\Users\kvidi\Desktop\whack24\ironyman\eyes\CP-AnemiC (A Conjunctival Pallor) Dataset from Ghana\CP-AnemiC dataset\Anemia_Data_Collection_Sheet.csv" 

dataset = EyesAnemiaDataset(image_dir=image_dir, csv_file=csv_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the ResNet50-based model
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        # Pretrained ResNet for image features
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer

        # Fully connected layers for classification and regression
        self.fc_class = nn.Sequential(
            nn.Linear(2048 + 1, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # Output for anemia classification (binary)
        )

        self.fc_regress = nn.Sequential(
            nn.Linear(2048 + 1, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # Output for HB_LEVEL regression
        )

    def forward(self, image, hb_level):
        # Extract image features
        image_features = self.resnet(image)

        # Concatenate image features with HB_LEVEL
        combined_features = torch.cat((image_features, hb_level.unsqueeze(1)), dim=1)

        # Predictions
        classification_output = self.fc_class(combined_features)
        regression_output = self.fc_regress(combined_features)

        return classification_output, regression_output


# Initialize the model
model = CombinedModel().to(device)

# Define loss functions
classification_criterion = nn.CrossEntropyLoss()  # For anemia classification
regression_criterion = nn.MSELoss()              # For HB_LEVEL prediction

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train_model():
    # Training loop
    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, hb_levels, labels in dataloader:
            images, hb_levels, labels = images.to(device), hb_levels.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            classification_output, regression_output = model(images, hb_levels)

            # Compute losses
            classification_loss = classification_criterion(classification_output, labels)
            regression_loss = regression_criterion(regression_output.squeeze(), hb_levels)
            total_loss = classification_loss + regression_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += total_loss.item()
            _, predicted = torch.max(classification_output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "eyes_anemia_hb_model.pth")


# For inference:
def predict(image):
    # Load model
    model.load_state_dict(torch.load("eyes_anemia_hb_model.pth"))
    model.eval()

    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)  

    # Forward pass
    with torch.no_grad():
        classification_output, regression_output = model(image, torch.tensor([0.0]).to(device))  # Dummy HB level for inference

    predicted_class = torch.argmax(classification_output, 1).item()
    predicted_hb_level = regression_output.item()

    # Return both the predicted class and HB level
    return predicted_class, predicted_hb_level


