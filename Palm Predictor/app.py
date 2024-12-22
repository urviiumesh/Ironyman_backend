from flask import Flask, request, jsonify
from PIL import Image
import tempfile
import os
from predictor import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    # Create a temporary file to store the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file_path = temp_file.name
        file.save(temp_file_path)  # Save the uploaded file to the temp path
    
    # Predict using the temporary file path
    result = predict(temp_file_path)
    
    # Optionally, delete the temp file after prediction
    os.remove(temp_file_path)
    
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
