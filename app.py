import google.generativeai as genai
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configure the Gemini API key (replace with your valid API key)
GENAI_API_KEY = "AIzaSyATkIjfp2GQy6lc45sYbN66UoZBMTrPBXY"
genai.configure(api_key=GENAI_API_KEY)

def get_anemia_diet_plan(anemia_status, cuisine, diet_type, allergens):
    # Construct the prompt
    prompt = f"""
    Generate an anemia diet plan for a person with anemia status: {'Yes' if anemia_status else 'No'}. 
    Consider the following preferences:
    - Cuisine: {cuisine}
    - Diet type: {diet_type}
    - Allergens: {', '.join(allergens) if allergens else 'None'}
    """

    # Initialize the Gemini model (using "gemini-1.5-flash" or another appropriate model)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    # Handle the response and extract the content
    print(response.text)
    return {"diet_plan": response.text} if response.text else {"error": "No diet plan generated."}

@app.route('/get_diet_plan', methods=['POST'])
def get_diet_plan_endpoint():
    data = request.get_json()

    # Extract relevant information from the request data
    anemia_status = data.get('anemia_status', False)
    cuisine = data.get('cuisine', 'Any')
    diet_type = data.get('diet_type', 'Any')
    allergens = data.get('allergens', [])

    # Generate the diet plan using Gemini API
    result = get_anemia_diet_plan(anemia_status, cuisine, diet_type, allergens)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
