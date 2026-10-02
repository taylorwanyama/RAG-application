import time


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}

    def allow(self, client_id: str) -> bool:
        now = time.monotonic()

        if client_id not in self.requests:
            self.requests[client_id] = {
                "window_start": now,
                "count": 1,
            }
            return True

        client = self.requests[client_id]

        if now - client["window_start"] >= self.window_seconds:
            client["window_start"] = now
            client["count"] = 1
            return True

        if client["count"] >= self.max_requests:
            return False

        client["count"] += 1
        return True