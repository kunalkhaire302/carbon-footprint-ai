from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from scripts.smoke_test import PAYLOAD


def predict(url: str) -> tuple[int, float]:
    if urllib.parse.urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError("Only HTTP(S) URLs are allowed")
    request = urllib.request.Request(
        url, method="POST", data=json.dumps(PAYLOAD).encode(), headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=15) as response:  # nosec B310
        response.read()
        return response.status, (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small, non-destructive prediction load check. Never target production without approval."
    )
    parser.add_argument("--base-url", default="http://localhost:5000")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.requests <= 500 or not 1 <= args.concurrency <= 32:
        raise SystemExit("requests must be 1..500 and concurrency 1..32")
    url = f"{args.base_url.rstrip('/')}/api/v1/predict"
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda _: predict(url), range(args.requests)))
    latencies = [duration for status, duration in results if status == 200]
    print(
        json.dumps(
            {
                "requests": len(results),
                "successes": len(latencies),
                "mean_ms": statistics.mean(latencies),
                "p95_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
