#!/usr/bin/env python3
"""
Send sample requests to a running prediction service and print the replies.

Start the service first (python predict.py, or the Docker container), then:
    python predict_test.py
    python predict_test.py --url http://localhost:9696
"""
import argparse
import json
import urllib.error
import urllib.request


def call(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9696")
    ap.add_argument("--request", default="sample_request.json")
    a = ap.parse_args()

    print("GET /health ->", call(f"{a.url}/health"))

    with open(a.request) as f:
        full = json.load(f)
    print("\nPOST /predict (all features) ->", call(f"{a.url}/predict", full))

    minimal = {k: full[k] for k in ("store_id", "product", "horizon", "month", "avail")}
    print("\nPOST /predict (required fields only) ->", call(f"{a.url}/predict", minimal))

    bad = dict(full, product="pineapples")
    print("\nPOST /predict (bad product, expect 400) ->", call(f"{a.url}/predict", bad))


if __name__ == "__main__":
    main()
