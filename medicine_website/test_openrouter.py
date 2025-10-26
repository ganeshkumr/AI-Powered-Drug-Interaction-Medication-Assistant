#!/usr/bin/env python3
"""
Quick test script to verify OpenRouter API integration
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openrouter_connection():
    """Test if OpenRouter API is configured and working"""
    
    print("=" * 60)
    print("OpenRouter API Connection Test")
    print("=" * 60)
    
    # Check if API key is set
    api_key = os.environ.get('OPENROUTER_API_KEY', '')
    
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        print("\nPlease set your API key in the .env file:")
        print("OPENROUTER_API_KEY=your_actual_key_here")
        return False
    
    if api_key == 'your_openrouter_api_key_here':
        print("❌ OPENROUTER_API_KEY is still set to placeholder value")
        print("\nPlease replace it with your actual API key from OpenRouter.ai")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    print("\nTesting connection to OpenRouter.ai...")
    
    # Test API call
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Medicine Assistant Test"
    }
    
    payload = {
        "model": "meta-llama/llama-3.2-3b-instruct:free",
        "messages": [
            {
                "role": "user",
                "content": "Say 'Hello! OpenRouter is working correctly.' in one sentence."
            }
        ],
        "temperature": 0.4,
        "max_tokens": 50
    }
    
    try:
        print("Sending test request...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        message = result['choices'][0]['message']['content']
        
        print("\n✅ SUCCESS! OpenRouter API is working correctly")
        print(f"\nAPI Response: {message}")
        print("\n" + "=" * 60)
        print("Your application is ready to use OpenRouter.ai!")
        print("=" * 60)
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 401:
            print("\n⚠️  Authentication failed - check your API key")
        elif response.status_code == 429:
            print("\n⚠️  Rate limit exceeded - wait a moment and try again")
        
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Connection Error: {e}")
        print("\nPossible issues:")
        print("- Check your internet connection")
        print("- Verify OpenRouter.ai is accessible")
        print("- Check firewall settings")
        return False
    
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        return False

def test_local_fallback():
    """Test if local LLM fallback is available"""
    
    print("\n" + "=" * 60)
    print("Testing Local LLM Fallback")
    print("=" * 60)
    
    api_url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "local-model",
        "messages": [{"role": "user", "content": "Test"}],
        "temperature": 0.4
    }
    
    try:
        print("Checking if local LLM server is running...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        print("✅ Local LLM server is available at localhost:1234")
        return True
    except:
        print("ℹ️  Local LLM server not running (this is OK if using OpenRouter)")
        return False

if __name__ == "__main__":
    print("\n")
    openrouter_ok = test_openrouter_connection()
    local_ok = test_local_fallback()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"OpenRouter API: {'✅ Working' if openrouter_ok else '❌ Not configured'}")
    print(f"Local LLM:      {'✅ Available' if local_ok else 'ℹ️  Not running'}")
    
    if openrouter_ok:
        print("\n🎉 Your application is ready to use OpenRouter.ai!")
        print("Run: python app.py")
    elif local_ok:
        print("\n⚠️  OpenRouter not configured, but local LLM is available")
        print("The app will use local LLM as fallback")
    else:
        print("\n⚠️  Neither OpenRouter nor local LLM is available")
        print("Please configure OpenRouter API key or start local LLM server")
    
    print("=" * 60 + "\n")
