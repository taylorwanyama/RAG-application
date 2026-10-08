import requests
import re

URL = "https://tenders.go.ke/build/assets/Tenders-e713f622.js"

response = requests.get(URL, timeout=30)

print("Status:", response.status_code)
print("Length:", len(response.text))

js = response.text

print("\n" + "=" * 80)
print("API / ENDPOINT CANDIDATES")
print("=" * 80)

# Look for strings containing likely API-related words
patterns = [
    r'["\'][^"\']*api[^"\']*["\']',
    r'["\'][^"\']*tender[^"\']*["\']',
    r'["\'][^"\']*(?:get|post|fetch|axios)[^"\']*["\']',
]

matches = set()

for pattern in patterns:
    for match in re.findall(pattern, js, re.IGNORECASE):
        matches.add(match)

for match in sorted(matches):
    print(match)


print("\n" + "=" * 80)
print("AXIOS / HTTP CODE")
print("=" * 80)

# Print sections containing axios calls
for match in re.finditer(r'axios', js, re.IGNORECASE):
    start = max(0, match.start() - 500)
    end = min(len(js), match.end() + 1000)

    print("\n" + "-" * 80)
    print(js[start:end])