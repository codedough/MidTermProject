"""Send sample requests to a running service (python serve.py, or the Docker container)."""
import json
import sys
import urllib.error
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696"


def call(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


print("GET /health ->", call("/health"))

with open("sample_request.json") as f:
    full = json.load(f)
cases = {
    "full request": full,
    "minimal request (5 required fields)": {"store_id": "S04", "product": "strawberries", "horizon": 2, "month": 9, "avail": 84.5},
    "store does not carry product (expect 422)": {"store_id": "S01", "product": "potatoes", "horizon": 1, "month": 5, "avail": 95},
    "availability out of range (expect 422)": {"store_id": "S01", "product": "apples", "horizon": 1, "month": 5, "avail": 140},
    "unknown field (expect 422)": {**full, "colour": "red"},
}
for name, payload in cases.items():
    status, body = call("/predict", payload)
    print(f"\n{name}\n  HTTP {status}: {json.dumps(body)[:300]}")
