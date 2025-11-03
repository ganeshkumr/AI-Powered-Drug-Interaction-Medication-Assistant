import requests

BASE_URL = "http://localhost:5000"

# First login
print("1. Logging in...")
login_response = requests.post(f"{BASE_URL}/api/login", json={
    "email": "test@example.com",
    "password": "Test@1234"
})
print(f"Login: {login_response.status_code}")

# Get session cookie
session = requests.Session()
session.cookies.update(login_response.cookies)

# Test medication check
print("\n2. Testing medication check...")
check_response = session.post(f"{BASE_URL}/check_before_adding", json={
    "drug_name": "Aspirin",
    "dosage_amount": "100",
    "dosage_unit": "mg",
    "frequency": "Once daily"
})

print(f"Status: {check_response.status_code}")
if check_response.status_code == 200:
    data = check_response.json()
    print(f"\nGNN Risk: {data.get('gnn_risk')}%")
    print(f"Verdict: {data.get('verdict')}")
    print(f"AI Response: {data.get('ai_response')[:200]}...")
    print(f"Can Add: {data.get('can_add')}")
else:
    print(f"Error: {check_response.text}")
