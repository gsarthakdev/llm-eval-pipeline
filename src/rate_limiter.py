import asyncio
import time

# Groq's free tier caps requests at 30/min and tokens at 8000/min for this
# model. Both the classifier and the LLM-judge hit the same account, so they
# share one gate that serializes calls with a minimum spacing between them.
MIN_INTERVAL_SECONDS = 3.0

_lock = asyncio.Lock()
_next_available_time = 0.0

async def throttle():
    """Block until it's safe to make another Groq request."""
    global _next_available_time
    async with _lock:
        now = time.monotonic()
        wait_time = _next_available_time - now
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            now = time.monotonic()
        _next_available_time = now + MIN_INTERVAL_SECONDS
