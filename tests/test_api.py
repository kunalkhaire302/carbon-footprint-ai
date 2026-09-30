from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "path",
    [
        "/health",
        "/ready",
        "/version",
        "/api/v1/health",
        "/api/v1/ready",
        "/api/v1/version",
        "/api/v1/metadata",
        "/api/v1/metrics",
    ],
)
def test_operational_endpoints(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_prediction_contract(client, valid_payload):
    response = client.post("/api/v1/predict", json=valid_payload, headers={"X-Request-ID": "contract-test"})
    body = response.get_json()
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "contract-test"
    assert body["prediction"]["total_tco2e_per_year"] == pytest.approx(7.0, abs=0.25)
    assert body["total_footprint_tco2e"] == body["prediction"]["total_tco2e_per_year"]
    assert body["model"]["version"] == "v1"
    assert body["prediction"]["confidence"] is None
    assert len(body["breakdown"]) == 6


def test_legacy_prediction_route(client, valid_payload):
    assert client.post("/predict", json=valid_payload).status_code == 200


def test_validation_error(client, valid_payload):
    valid_payload["internet_usage_hours"] = 25
    response = client.post("/api/v1/predict", json=valid_payload)
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "VALIDATION_ERROR"


def test_malformed_json(client):
    response = client.post("/api/v1/predict", data="{", content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "MALFORMED_JSON"


def test_wrong_content_type(client):
    response = client.post("/api/v1/predict", data="payload", content_type="text/plain")
    assert response.status_code == 415


def test_invalid_request_id_is_replaced(client):
    response = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert response.headers["X-Request-ID"] != "bad id with spaces"


def test_removed_sensitive_endpoints(client):
    assert client.get("/history").status_code == 404
    assert client.post("/retrain").status_code in {404, 405}
