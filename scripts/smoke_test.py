from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request

PAYLOAD = {
    "household_size": 2,
    "electricity_usage_kwh": 250,
    "heating_source": "none",
    "vehicle_type": "petrol",
    "vehicle_km": 800,
    "flights_short_haul": 1,
    "flights_long_haul": 0,
    "diet_type": "non-vegetarian",
    "grocery_spend_monthly": 300,
    "waste_kg_weekly": 15,
    "internet_usage_hours": 6,
}


def request_json(url: str, method: str = "GET", payload: object | None = None) -> tuple[int, dict[str, object]]:
    if urllib.parse.urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError("Only HTTP(S) URLs are allowed")
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, method=method, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # nosec B310
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def run_remote(base_url: str) -> None:
    check(request_json(f"{base_url}/health")[0] == 200, "Health check failed")
    check(request_json(f"{base_url}/ready")[0] == 200, "Readiness check failed")
    status, prediction = request_json(f"{base_url}/api/v1/predict", "POST", PAYLOAD)
    check(status == 200 and "prediction" in prediction and "metadata" in prediction, "Prediction contract failed")
    malformed = urllib.request.Request(
        f"{base_url}/api/v1/predict", method="POST", data=b"{", headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(malformed, timeout=10)  # nosec B310
    except urllib.error.HTTPError as error:
        check(error.code == 400, "Malformed JSON did not return 400")
    else:
        raise AssertionError("Malformed JSON was accepted")


def run_local() -> None:
    from backend.app import create_app
    from backend.app.config import TestSettings

    with create_app(TestSettings.from_env()).test_client() as client:
        check(client.get("/").status_code == 200, "Frontend failed")
        check(client.get("/health").status_code == 200, "Health check failed")
        check(client.get("/ready").status_code == 200, "Readiness check failed")
        response = client.post("/api/v1/predict", json=PAYLOAD)
        check(response.status_code == 200 and "prediction" in response.get_json(), "Prediction contract failed")
        check(
            client.post("/api/v1/predict", data="{", content_type="application/json").status_code == 400,
            "Malformed JSON did not return 400",
        )


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:5000")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    run_local() if args.local else run_remote(args.base_url.rstrip("/"))
    print("Smoke test passed")


if __name__ == "__main__":
    main()
