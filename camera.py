import pytesseract
import re
from flask import Flask, request, render_template, jsonify
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

# Route for the homepage
@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prescription Reader</title>
    </head>
    <body>
        <h1>Prescription Reader</h1>
        <form action="/process_image" method="POST" enctype="multipart/form-data">
            <label for="image">Upload Prescription Image:</label><br>
            <input type="file" name="image" id="image" accept="image/*" required><br><br>
            <button type="submit">Process Image</button>
        </form>
    </body>
    </html>
    """

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

        # Clean up the uploaded file (optional)
        os.remove(file_path)

        # Return the cleaned text as HTML
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Processed Prescription</title>
        </head>
        <body>
            <h1>Processed Prescription</h1>
            <p><strong>Extracted and Cleaned Text:</strong></p>
            <pre>{cleaned_text}</pre>
            <a href="/">Go Back</a>
        </body>
        </html>
        """
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
