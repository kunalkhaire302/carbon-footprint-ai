from backend.app.domain.benchmarks import INDIA_AVG, WORLD_AVG


def calculate_grade(total_tco2e: float) -> tuple[int, str]:
    india = INDIA_AVG
    world = WORLD_AVG
    z_score = (total_tco2e - india) / 1.5
    percentile = max(min(round(100 - min(max(z_score, 0), 3) / 3 * 100), 99), 1)
    if total_tco2e < india:
        grade = "A"
    elif total_tco2e < india * 1.5:
        grade = "B"
    elif total_tco2e < world:
        grade = "C"
    elif total_tco2e < round(world * 1.5, 10):
        grade = "D"
    else:
        grade = "F"
    return int(percentile), grade
