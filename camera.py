import pytesseract
import re
from flask import Flask, request, jsonify
from PIL import Image
import os

# Set the path to Tesseract executable (if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize Flask app
app = Flask(__name__)

# Create a folder to store uploaded images temporarily
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to extract text from an image using OCR
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Function to block personal information using regular expressions
def block_personal_information(text):
    patterns = {
        "phone_number": r"\+?\(?\d{1,4}[\)\-]?\s?\(?\d{1,4}\)?[\s\-]?\d{1,4}[\s\-]?\d{1,4}[\s\-]?\d{1,4}",  # Matches phone numbers
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Matches email addresses
        "ssn": r"\d{3}-\d{2}-\d{4}",  # Matches social security numbers
        "credit_card": r"\b(?:\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}|\d{4}[\s\-]?\d{6}[\s\-]?\d{5})\b",  # Matches credit card numbers
    }
    for key, pattern in patterns.items():
        text = re.sub(pattern, f"[REDACTED {key.upper()}]", text)
    return text

# Function to extract only the medicine names
def extract_medicine_names(text):
    """
    Extracts the medicine names from the given text.
    """
    # Regular expression to match medicine names
    pattern = r"^[A-Za-z\s]+(?=\s\d+\.\d+gm)"
    matches = re.findall(pattern, text, re.MULTILINE)
    medicine_names = [match.strip() for match in matches]
    return medicine_names

# Route to process the uploaded image
@app.route("/process_image", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return "No file uploaded!", 400

    file = request.files["image"]
    if file.filename == "":
        return "No file selected!", 400

    # Save the uploaded image to the uploads folder
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        # Extract text from the uploaded image
        extracted_text = extract_text_from_image(file_path)

        # Block personal information in the extracted text
        cleaned_text = block_personal_information(extracted_text)

        # Extract medicine names
        medicine_names = extract_medicine_names(cleaned_text)

        # Clean up the uploaded file (optional)
        os.remove(file_path)

        # Return the medicine names as plain text
        return jsonify({"medicine_names": medicine_names})

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True,port=6001)
