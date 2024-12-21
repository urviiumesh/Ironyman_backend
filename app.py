from flask import Flask, request, jsonify
import requests
import json

app = Flask(__name__)

# Replace with your actual Gemini API key (store securely)
GEMINI_API_KEY = "..."

# Function to interact with the Gemini API
def get_anemia_diet_plan(anemia_status, cuisine, diet_type, allergens):
    """
    Fetches a customized anemia diet plan from the Gemini API.

    Args:
        anemia_status: True if the person has anemia, False otherwise.
        cuisine: Desired cuisine (e.g., "Indian", "Mexican", "Mediterranean").
        diet_type: "vegetarian", "vegan", "pescatarian", "gluten-free", etc.
        allergens: List of allergens (e.g., "nuts", "dairy", "seafood").

    Returns:
        A dictionary containing the generated diet plan.
    """
    prompt = f"""
    Generate an anemia diet plan for a person with anemia status: {anemia_status}. 
    Consider the following preferences:
    * Cuisine: {cuisine}
    * Diet type: {diet_type}
    * Allergens: {allergens}

    The diet plan should include:
    * A list of recommended foods rich in iron, vitamin B12, and folate.
    * Sample meal ideas incorporating these foods.
    * Tips for maximizing iron absorption. 
    """

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": prompt,
        "max_tokens": 500,  # Adjust as needed
        "temperature": 0.7  # Adjust for creativity
    }

    try:
        response = requests.post("https://api.gemini.google.com/v1/chat", headers=headers, json=data)
        response.raise_for_status()

        response_json = response.json()
        diet_plan = response_json["generated_text"]

        # Optional parsing to extract specific details (example)
        recommended_foods = extract_foods(diet_plan)  # Implement this function

        return jsonify({'diet_plan': diet_plan, 'recommended_foods': recommended_foods})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Implement a function to parse the diet plan text (optional)
def extract_foods(diet_plan):
    # ... (logic to extract food items from the text)
    # ...
    return extracted_foods

@app.route('/get_diet_plan', methods=['POST'])
def get_diet_plan_endpoint():
    """
    API endpoint to handle diet plan requests.
    """
    data = request.get_json()
    anemia_status = data.get('anemia_status', False)
    cuisine = data.get('cuisine', 'Any')
    diet_type