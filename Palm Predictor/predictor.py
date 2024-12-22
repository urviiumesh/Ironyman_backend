from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

# Define the label map
label_map = {0: 'Non-Anemic', 1: 'Anemic'}

# Define the same transformations used during training
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained model
model = models.resnet50(pretrained=False)  # Start with the base architecture
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Adjust for 2 classes
model.load_state_dict(torch.load(r"C:\Users\kvidi\Desktop\whack24\ironyman\Palm Predictor\best_model.pth", map_location=torch.device('cpu')))  # Load the weights
model.eval()  # Set to evaluation mode

def predict(image_path):
    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = data_transforms(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)  # Pass the image through the model
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
        predicted_label = label_map[predicted.item()]  # Map to the label

    return predicted_label
