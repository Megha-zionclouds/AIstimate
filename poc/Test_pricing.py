import requests
import json

# Step 1: Replace with your actual API key
API_KEY = "emym6vnmxyo0d62vz3w1kaj"

# Step 2: Set the base URL (use sandbox for testing)
BASE_URL = "https://api.clearestimates.com/cepia/"


# Step 3: Test authentication
def test_auth():
    url = BASE_URL
    headers = {"x-api-key": API_KEY}
    response = requests.post(url, headers=headers)
    print("AUTH TEST:", response.status_code, response.json())

# Step 4: Test scope search
def test_scope_search():
    url = BASE_URL + "search/"
    headers = {"x-api-key": API_KEY}
    payload = {"query": "trim"}
    response = requests.post(url, headers=headers, json=payload)
    print("SEARCH TEST:", response.status_code)
    print(json.dumps(response.json(), indent=2))

# Step 5: Create estimate (Level 1)
def test_create_estimate():
    url = BASE_URL + "estimates/create"
    headers = {"x-api-key": API_KEY}
    payload = {
        "zipcode": "80210",
        "estimatearray": [
            {"scope_id": "3e380c89-80a5-4512-ae50-0e4fd7c7d2c7", "quantity": 200},
            {"scope_id": "c0f8198d-ce99-4b55-985a-101170415d1f", "quantity": 8}
        ],
        "type": 1
    }
    response = requests.post(url, headers=headers, json=payload)
    print("ESTIMATE TEST:", response.status_code)
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_auth()
    test_scope_search()
    test_create_estimate()
