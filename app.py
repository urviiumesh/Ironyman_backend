from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

GEMINI_API_KEY = "AIzaSyATkIjfp2GQy6lc45sYbN66UoZBMTrPBXY"

@app.route('/')
def home():
    return (
        "Welcome to the Anemia Diet Plan API!<br>"
        "Use the <b>/get_diet_plan</b> endpoint with a POST request to generate a diet plan.<br>"
        "Example payload: <br>"
        '{"anemia_status": true, "cuisine": "Indian", "diet_type": "vegetarian", "allergens": ["nuts", "dairy"]}'
    )

@app.route('/get_diet_plan', methods=['GET'])
def get_diet_plan_test():
    return "This endpoint only supports POST requests. Use POST to submit data."

def get_anemia_diet_plan(anemia_status, cuisine, diet_type, allergens):
    prompt = f"""
    Generate an anemia diet plan for a person with anemia status: {'Yes' if anemia_status else 'No'}. 
    Consider the following preferences:
    - Cuisine: {cuisine}
    - Diet type: {diet_type}
    - Allergens: {', '.join(allergens) if allergens else 'None'}
    """
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0.7
    }
    try:
        response = requests.post("https://api.gemini.google.com/v1/chat", headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        diet_plan = response_json.get("generated_text", "No diet plan generated.")
        return {"diet_plan": diet_plan}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

@app.route('/get_diet_plan', methods=['POST'])
def get_diet_plan_endpoint():
    try:
        data = request.get_json()
        anemia_status = data.get('anemia_status', False)
        cuisine = data.get('cuisine', 'Any')
        diet_type = data.get('diet_type', 'Any')
        allergens = data.get('allergens', [])
        result = get_anemia_diet_plan(anemia_status, cuisine, diet_type, allergens)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
