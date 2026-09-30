INDIA_AVG = 1.9
WORLD_AVG = 4.7

BENCHMARKS: dict[str, dict[str, object]] = {
    "india": {
        "value_tco2e_per_year": INDIA_AVG,
        "geography": "India",
        "year": None,
        "source": "legacy application assumption; authoritative source review required",
    },
    "global": {
        "value_tco2e_per_year": WORLD_AVG,
        "geography": "Global",
        "year": None,
        "source": "legacy application assumption; authoritative source review required",
    },
}
