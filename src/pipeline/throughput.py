"""Average rates from total completed work and measured seconds."""


def rate(count: int, seconds: float, unit: str, window: str) -> dict:
    return {
        "count": count,
        "seconds": seconds,
        "unit": unit,
        "window": window,
        "per_second": count / seconds if seconds > 0 else None,
    }
