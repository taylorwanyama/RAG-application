import requests
import re

url = "https://tenders.go.ke/build/assets/app-eddcbfe0.js"

response = requests.get(url, timeout=30)

print("Status:", response.status_code)
print("Length:", len(response.text))

js = response.text

# Look for URLs / API-looking strings
patterns = [
    r'https?://[^"\']+',
    r'["\'][^"\']*(?:api|tenders)[^"\']*["\']',
]

for pattern in patterns:
    print("\n--- MATCHES ---")

    matches = re.findall(pattern, js, re.IGNORECASE)

    for match in matches[:100]:
        print(match)