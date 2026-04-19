#!/usr/bin/env python3
"""
run_cloud_ingestion.py — Push BOM data to cognee Cloud and cognify.

Usage:
    python3 run_cloud_ingestion.py              # upload all parts + cognify
    python3 run_cloud_ingestion.py --cognify    # only cognify (files already uploaded)
    python3 run_cloud_ingestion.py --search     # only test search
"""

import argparse
import os
import sys
import time
import glob
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("COGNEE_API_KEY")
BASE = "https://tenant-d90a4d3d-cbba-4d05-a592-147fa1cc5aa6.aws.cognee.ai/api"
HEADERS_JSON = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
HEADERS_AUTH = {"X-Api-Key": API_KEY}
DATASET = "agnes_bom"


def upload_file(filepath):
    """Upload a single file to cognee Cloud."""
    name = os.path.basename(filepath)
    print(f"  Uploading {name}...")
    with open(filepath, "rb") as f:
        r = requests.post(
            f"{BASE}/v1/add",
            headers=HEADERS_AUTH,
            files={"data": (name, f, "text/plain")},
            data={"datasetName": DATASET},
            timeout=120,
        )
    if r.status_code == 200:
        print(f"  OK: {name}")
        return True
    else:
        print(f"  FAIL ({r.status_code}): {r.text[:200]}")
        return False


def cognify():
    """Run cognify on the dataset (fire-and-forget, then poll)."""
    print("Triggering cognify (fire-and-forget)...")
    try:
        r = requests.post(
            f"{BASE}/v1/cognify",
            headers=HEADERS_JSON,
            json={"datasets": [DATASET]},
            timeout=10,  # short timeout — just trigger it
        )
        print(f"  Triggered: {r.status_code}")
    except requests.exceptions.ReadTimeout:
        print("  Triggered (server processing in background)")
    except Exception as e:
        print(f"  Trigger sent (may be processing): {e}")

    # Poll for completion
    print("  Waiting for cognify to complete (checking every 30s)...")
    for attempt in range(20):  # up to 10 minutes
        time.sleep(30)
        try:
            r = requests.post(
                f"{BASE}/v1/search",
                headers=HEADERS_JSON,
                json={"query_text": "vitamin d3", "query_type": "CHUNKS", "datasets": [DATASET]},
                timeout=30,
            )
            if r.status_code == 200:
                results = r.json()
                for item in results:
                    sr = item.get("search_result", [])
                    if sr and "No" not in str(sr[0])[:20] and "not" not in str(sr[0])[:20].lower():
                        print(f"  Cognify complete! Search returned data after {(attempt+1)*30}s")
                        return True
            print(f"  Still processing... ({(attempt+1)*30}s)")
        except Exception:
            print(f"  Still processing... ({(attempt+1)*30}s)")

    print("  Cognify may still be running. Try --search later.")
    return False


def search(query="vitamin d3 suppliers"):
    """Test search."""
    print(f"Searching: '{query}'...")
    r = requests.post(
        f"{BASE}/v1/search",
        headers=HEADERS_JSON,
        json={"query_text": query, "query_type": "CHUNKS", "datasets": [DATASET]},
        timeout=60,
    )
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        results = r.json()
        for item in results:
            ds = item.get("dataset_name", "")
            sr = item.get("search_result", [])
            print(f"  [{ds}] {str(sr)[:300]}")
    else:
        print(f"  Error: {r.text[:200]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognify", action="store_true")
    parser.add_argument("--search", action="store_true")
    args = parser.parse_args()

    if args.search:
        search()
        return

    if args.cognify:
        cognify()
        search()
        return

    # Full run: upload all parts then cognify
    parts = sorted(glob.glob("assets/agnes_bom_part*.txt"))
    if not parts:
        print("ERROR: No assets/agnes_bom_part*.txt files found. Run 'python3 run_cloud_ingestion.py' after migrating BOM data.")
        sys.exit(1)

    print(f"Uploading {len(parts)} files to cognee Cloud dataset '{DATASET}'...")
    for filepath in parts:
        ok = upload_file(filepath)
        if not ok:
            print(f"Upload failed for {filepath}. Stopping.")
            sys.exit(1)
        time.sleep(2)  # rate limit buffer

    print(f"\nAll {len(parts)} files uploaded.")
    cognify()
    print("\nTesting search...")
    search()
    print("\nDone! Data is now in cognee Cloud. Dify can search it.")


if __name__ == "__main__":
    main()
