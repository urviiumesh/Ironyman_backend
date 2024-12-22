import requests
from geopy.distance import geodesic
from flask import Flask, jsonify

# Constants
IPSTACK_API_KEY = "601ec84c00329d73d8f4fd795a2e8661"  # Replace with your IPStack API key
OVERPASS_API_URL = "http://overpass-api.de/api/interpreter"

app = Flask(__name__)

# Function to get the user's current location using IPStack
def get_user_location():
    try:
        response = requests.get(f"http://api.ipstack.com/check?access_key={IPSTACK_API_KEY}")
        if response.status_code == 200:
            data = response.json()
            return (data["latitude"], data["longitude"])
        else:
            return {"error": "Failed to fetch user location. Check your IPStack API key or internet connection."}
    except Exception as e:
        return {"error": f"Error fetching user location: {e}"}

# Function to fetch Jan Aushadhi Kendra locations using Overpass API
def fetch_jan_aushadi_locations(user_coords):
    try:
        overpass_query = f"""
        [out:json];
        node["name"~"Jan Aushadhi Kendra"](around:50000,{user_coords[0]},{user_coords[1]});
        out body;
        """
        response = requests.get(OVERPASS_API_URL, params={"data": overpass_query})
        if response.status_code == 200:
            return response.json().get("elements", [])
        else:
            return {"error": "Error fetching data from Overpass API."}
    except Exception as e:
        return {"error": f"Error fetching Jan Aushadhi locations: {e}"}

# Function to find the nearest branch
def find_nearest_branch(user_coords, branches):
    nearest = None
    min_distance = float("inf")
    for branch in branches:
        branch_coords = (branch["lat"], branch["lon"])
        distance = geodesic(user_coords, branch_coords).kilometers
        if distance < min_distance:
            min_distance = distance
            nearest = branch
    return nearest, min_distance

@app.route("/nearest_kendra", methods=["GET"])
def nearest_kendra():
    """Endpoint to return nearest Jan Aushadhi Kendra."""
    user_coords = get_user_location()
    if isinstance(user_coords, dict) and "error" in user_coords:
        return jsonify(user_coords), 400  # Return the error message if fetching location fails

    branches = fetch_jan_aushadi_locations(user_coords)
    if isinstance(branches, dict) and "error" in branches:
        return jsonify(branches), 400  # Return the error message if fetching branches fails

    if branches:
        nearest_branch, distance = find_nearest_branch(user_coords, branches)
        if nearest_branch:
            return jsonify({
                "nearest_branch": {
                    "name": nearest_branch.get('tags', {}).get('name', 'Unknown'),
                    "coordinates": {
                        "lat": nearest_branch['lat'],
                        "lon": nearest_branch['lon']
                    },
                    "distance_km": round(distance, 2)
                }
            }), 200
        else:
            return jsonify({"message": "No branches found nearby."}), 404
    else:
        return jsonify({"message": "No branches found."}), 404


if __name__ == "_main_":
    app.run(debug=True, port=6003)