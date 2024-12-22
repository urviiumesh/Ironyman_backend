from flask import Flask, request, jsonify
import requests
import os
import json

app = Flask(__name__)

# Create a folder to store uploaded images temporarily
UPLOAD_FOLDER = r"C:\Users\kvidi\Desktop\whack24\ironyman\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def get_generic_name_from_brand_name(medicine_name):
    """Fetches the generic name of a given medicine brand using the openFDA API."""
    search_term = medicine_name.replace(" ", "+")  # Prepare for URL
    url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:\"{search_term}\""

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            generic_names = []
            for result in data["results"]:
                if "openfda" in result and "generic_name" in result["openfda"]:
                    generic_names.extend(result["openfda"]["generic_name"])
            if generic_names:
                return generic_names
            else:
                print(f"No generic names found for '{medicine_name}'.")
                return None
        else:
            print(f"No results found for brand name '{medicine_name}'.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}. Response text: {response.text}")
        return None

@app.route('/process_generic_names', methods=['POST'])
def process_generic_names():
    """Accepts an image or text, processes it for medicine names, and fetches the corresponding generic names."""
    # Case 1: Text-based input for medicine names
    if request.is_json:
        data = request.get_json()
        medicine_names = data.get("medicine_names", [])
        if not medicine_names:
            return jsonify({"error": "No medicine names provided"}), 400
    # Case 2: Image-based input (file upload)
    elif 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected!"}), 400

        # Save the uploaded image to the uploads folder
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        try:
            # Send the image to the external API to process it and extract medicine names
            external_api_url = "http://127.0.0.1:6001/process_image"
            with open(file_path, "rb") as image_file:
                external_response = requests.post(external_api_url, files={"image": image_file})

            external_response.raise_for_status()  # Check for errors in the external API
            external_data = external_response.json()

            # Extract medicine names from the response
            medicine_names = external_data.get("medicine_names", [])
            if not medicine_names:
                return jsonify({"error": "No medicine names provided by external API"}), 400

        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Error calling external API: {str(e)}"}), 500
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Error decoding JSON response from external API: {str(e)}"}), 500
        except Exception as e:
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    else:
        return jsonify({"error": "No valid input provided (either JSON or file upload required)"}), 400

    # Step 3: Fetch the generic names for each medicine
    result = {}
    for medicine_name in medicine_names:
        print(f"\nSearching for generic name for: {medicine_name}")
        generic_names = get_generic_name_from_brand_name(medicine_name)
        result[medicine_name] = generic_names or "No generic names found"

    # Clean up the uploaded file (optional)
    if 'image' in request.files:
        os.remove(file_path)

    return jsonify(result), 200

if __name__ == "__main__":
    app.run(port=6002, debug=True)
