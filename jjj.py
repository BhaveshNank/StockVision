import os
import requests

API_KEY = os.getenv("FINNHUB_API_KEY")
print(f"🔑 API Key being used: {API_KEY}")  # Debugging step

if not API_KEY:
    print("❌ API Key is missing!")
else:
    url = f"https://finnhub.io/api/v1/company-news?symbol=AAPL&from=2024-02-10&to=2024-02-17&token={API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        print("✅ API Key is working! News data received.")
    else:
        print(f"❌ API Error {response.status_code} - {response.text}")
