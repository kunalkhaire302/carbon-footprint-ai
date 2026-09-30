from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from backend.app.errors import ApiError


class RateLimiter:
    def __init__(self, window_seconds: int) -> None:
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                raise ApiError("RATE_LIMITED", "Too many requests; try again later", 429)
            events.append(now)
