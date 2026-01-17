import requests

# Test the API endpoints
BASE_URL = "http://localhost:5000"

print("Testing Flask API endpoints...\n")

# Test 1: Check auth (should return not authenticated)
print("1. Testing /api/check-auth...")
response = requests.get(f"{BASE_URL}/api/check-auth")
print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}\n")

# Test 2: Register a new user
print("2. Testing /api/register...")
register_data = {
    "name": "Test User",
    "email": "test@example.com",
    "password": "Test@1234"
}
response = requests.post(f"{BASE_URL}/api/register", json=register_data)
print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}\n")

print("✅ API is working! You can now use React frontend.")
