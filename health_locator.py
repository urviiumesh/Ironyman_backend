import requests
from geopy.distance import geodesic

# Constants
IPSTACK_API_KEY = "601ec84c00329d73d8f4fd795a2e8661"  # Replace with your IPStack API key
OVERPASS_API_URL = "http://overpass-api.de/api/interpreter"

# Function to get the user's current location using IPStack
def get_user_location():
    try:
        response = requests.get(f"http://api.ipstack.com/check?access_key={IPSTACK_API_KEY}")
        if response.status_code == 200:
            data = response.json()
            return (data["latitude"], data["longitude"])
        else:
            print("Failed to fetch user location. Check your IPStack API key or internet connection.")
            return None
    except Exception as e:
        print(f"Error fetching user location: {e}")
        return None

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
            print("Error fetching data from Overpass API.")
            return []
    except Exception as e:
        print(f"Error fetching Jan Aushadhi locations: {e}")
        return []

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

# Main program
if __name__ == "__main__":
    print("Locating your nearest Jan Aushadhi Kendra...")
    user_coords = get_user_location()
    if user_coords:
        branches = fetch_jan_aushadi_locations(user_coords)
        if branches:
            nearest_branch, distance = find_nearest_branch(user_coords, branches)
            if nearest_branch:
                print(f"\nNearest Jan Aushadhi Kendra:\n"
                      f"Name: {nearest_branch.get('tags', {}).get('name', 'Unknown')}\n"
                      f"Coordinates: {nearest_branch['lat']}, {nearest_branch['lon']}\n"
                      f"Distance: {distance:.2f} km\n")
            else:
                print("No branches found nearby.")
        else:
            print("No branches found.")
    else:
        print("Unable to determine your location.")
