import requests
import json

BASE_URL = "http://localhost:8000"
API_KEY = "mozhil-api-key-2024"

def test_health():
    print("Testing /health ...", end=" ")
    resp = requests.get(f"{BASE_URL}/health")
    if resp.status_code == 200:
        print("PASS")
        print(f"Response: {resp.json()}")
    else:
        print(f"FAIL ({resp.status_code})")
        print(resp.text)

def test_unauthorized():
    print("Testing Unauthorized Access ...", end=" ")
    headers = {"x-api-key": "wrong-key", "Content-Type": "application/json"}
    data = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": "dGVzdA=="
    }
    resp = requests.post(f"{BASE_URL}/api/voice-detection", headers=headers, json=data)
    if resp.status_code == 401:
        print("PASS")
        print(f"Response: {resp.json()}")
    else:
        print(f"FAIL ({resp.status_code})")
        print(resp.text)

def test_validation_error():
    print("Testing Validation Error (missing fields) ...", end=" ")
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    data = {
        "language": "English",
        "audioFormat": "mp3"
        # audioBase64 missing
    }
    resp = requests.post(f"{BASE_URL}/api/voice-detection", headers=headers, json=data)
    if resp.status_code == 400 or resp.status_code == 422:
        print("PASS")
        print(f"Response: {resp.json()}")
    else:
        print(f"FAIL ({resp.status_code})")
        print(resp.text)

if __name__ == "__main__":
    test_health()
    test_unauthorized()
    test_validation_error()
