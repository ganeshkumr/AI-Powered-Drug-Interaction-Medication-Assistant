import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get('OPENROUTER_API_KEY', '')

print("Testing different OpenRouter free models...\n")

# List of free models to try
models = [
    "mistralai/mistral-7b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "microsoft/phi-3-mini-128k-instruct:free"
]

test_prompt = "Say 'Hello, I am working!' in one sentence."

for model in models:
    print(f"Testing: {model}")
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 50
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()['choices'][0]['message']['content']
            print(f"✅ SUCCESS: {result[:50]}...\n")
        else:
            print(f"❌ FAILED: {response.status_code} - {response.text[:100]}\n")
    except Exception as e:
        print(f"❌ ERROR: {str(e)[:100]}\n")

print("\nRecommendation: Use the model that succeeded above!")
