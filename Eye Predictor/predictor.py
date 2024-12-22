from flask import Flask, request, jsonify
import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
from werkzeug.utils import secure_filename
from model_eye import CombinedModel  # Assuming your model class is defined in model_eye.py

# Flask app setup
app = Flask(__name__)

# Define allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Define device (cuda if available, otherwise cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing transform (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resizing to 224x224 for ResNet
    transforms.ToTensor(),           # Converting image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
])

# Load the model
model = CombinedModel().to(device)

# Make sure model is loaded correctly for inference (only loading weights, no training)
model.load_state_dict(torch.load(r"C:\Users\kvidi\Desktop\whack24\ironyman\eyes_anemia_hb_model.pth", map_location=device))
model.eval()  # Set the model to evaluation mode to avoid training layers being applied

# Function to extract the bottom part of the eye
def extract_eyelid(image_path):
    try:
        # 1. Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # 2. Define a fixed region for the bottom part of the eye
        height, width, _ = img.shape
        crop_height = int(height * 0.25)  # From the middle to the bottom part (adjust as necessary)
        crop_bottom = height  # Full height
        crop_top = crop_height  # Start cropping from the 60% height of the image
        crop_left = int(width * 0.3)  # 30% from the left edge (you can adjust for better centering)
        crop_right = int(width * 0.7)  # 70% from the left edge (you can adjust for better centering)
        
        # Crop the image to focus on the bottom part of the eye
        cropped_eyelid = img[crop_top:crop_bottom, crop_left:crop_right]

        return cropped_eyelid

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Prediction function
def predict(image_path):
    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to the device (GPU or CPU)

    # Perform the forward pass and get predictions
    with torch.no_grad():  # No gradients should be computed during inference
        classification_output, regression_output = model(image, torch.tensor([0.0]).to(device))  # Dummy HB level

    # Get the predicted class (0 for not anemic, 1 for anemic)
    predicted_class = torch.argmax(classification_output, 1).item()

    # Get the predicted HB level
    predicted_hb_level = regression_output.item()

    # Return the results
    return predicted_class, predicted_hb_level

# Define a helper function to check allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask route for image upload and prediction
@app.route('/predict', methods=['POST'])
def upload_image():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']

    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the file has a valid extension
    if file and allowed_file(file.filename):
        # Secure the filename and save it temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file.save(file_path)

        # Step 1: Extract eyelid
        cropped_image = extract_eyelid(file_path)

        if cropped_image is not None:
            # Step 2: Save the cropped image temporarily
            cropped_image_path = os.path.join('uploads', 'cropped_eyelid.jpg')
            cv2.imwrite(cropped_image_path, cropped_image)

            # Step 3: Make prediction
            predicted_class, predicted_hb_level = predict(cropped_image_path)

            # Step 4: Return the result
            result = {
                'prediction': 'Anemic' if predicted_hb_level<=12 else 'Not Anemic',
                'predicted_hb_level': predicted_hb_level
            }

            # Clean up: Delete the cropped image after prediction
            os.remove(file_path)
            os.remove(cropped_image_path)

            return jsonify(result)

        else:
            # Handle error if the eyelid extraction fails
            return jsonify({'error': 'Failed to extract eyelid from the image'}), 500

    return jsonify({'error': 'Invalid file format'}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
