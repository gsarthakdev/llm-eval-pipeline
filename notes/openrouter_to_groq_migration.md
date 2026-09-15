# From OpenAI to Groq: a free-tier migration story

**Date:** 09-15-2026

My OpenAI credits expired mid-project :(

The eval pipeline (51 golden test cases, each needing a classification call plus an LLM-as-judge call) needed a new home, and it needed to be free. Here's how I actually went about that, including the dead end I walked into first.

## OpenRouter looked like the answer, then wasn't

OpenRouter was the obvious first stop. One API key, dozens of providers, a genuinely free tier with no credit card required. I swapped the client's `base_url` and pointed it at `nvidia/nemotron-3-ultra-550b-a55b:free`, expecting a five-minute fix.

It wasn't. Running the full 51-case batch crashed immediately with a `TypeError` buried deep in the OpenAI SDK's response parser. Digging in, the real cause was uglier than a normal error: OpenRouter's free Nvidia-hosted model was returning HTTP 200 with an error payload instead of real choices whenever it got overloaded, which under 51 concurrent requests was often. The SDK had no idea what to do with "success" that wasn't actually success.

I could work around that. Added retry with backoff, checked for the error payload before trusting the response, capped concurrency. Reran it. New failure, and this one was an honest error:

```
Rate limit exceeded: free-models-per-day.
Add 10 credits to unlock 1000 free model requests per day.
```

That was the actual problem. OpenRouter's free `:free` models aren't just rate-limited, they're capped at 50 requests a day, account-wide. My batch needed roughly 102 requests (51 cases times 2 calls each) just to run once. Even with perfect retry logic I'd only get about halfway through a single run before getting locked out until the next day. That's not something I could fix with better code. It's a ceiling.

The retry logic and error-body checking I'd just written for OpenRouter was correct, but it was solving the wrong problem. The overload errors were real, but they sat on top of a daily quota that no amount of client-side handling gets past. At that point the right move wasn't to write more retry code, it was to stop and ask whether OpenRouter was viable at all.

## Weighing the alternatives

I laid out the actual options instead of just grabbing the next thing that came to mind.

| Option | Why not |
|---|---|
| Wait about 10 hours for OpenRouter's daily reset | Free, but this problem comes back every single day the eval runs |
| Pay $10 for OpenRouter credits | Defeats the point of a free eval pipeline, and just raises a self-imposed ceiling |
| Try a different model on OpenRouter | Same account, same 50/day cap, doesn't help |
| Groq | Free tier too, but capped per minute instead of per day. Different shape of limit entirely |

I checked with the user before deciding between paid and free rather than assuming. They confirmed: stay free, find another provider. Groq was already sitting in the codebase, half considered, a commented out client line from an earlier experiment and a `GROQ_API_KEY` already sitting in `.env`. Somebody had scoped this out before and never finished it, so it made sense to pick that thread back up instead of starting cold.

Before rewiring anything I did one quick sanity check: a single real request against Groq's `openai/gpt-oss-20b` model with structured output, to confirm it could actually do what the pipeline needed. It worked cleanly.

## Why Groq's limit actually held up

Groq's free tier caps at 30 requests per minute, plus a token budget alongside it, same general idea as OpenRouter. The difference that mattered wasn't the number, it was that Groq's clock resets every 60 seconds instead of every 24 hours. A per-minute cap is something you can just wait out mid-run. A per-day cap ends your run for the day. That's really the whole reason this move was viable at all, even though both providers advertise "a free rate limit" on paper.

Switching providers meant hitting Groq's limit blind the first time too. The first full run after the switch died with a 429 almost right away, because my old concurrency cap only controlled how many requests were in flight at once, not how fast they went out. Those are two different knobs and I'd only turned one of them. The fix was pacing every request through a single shared throttle so the classifier calls and judge calls both stay under Groq's per-minute ceiling, no matter how many test cases are running at once. After that, the full 51-case batch ran end to end in about five minutes.

## The actual throttle, and why 3 seconds isn't a random number

Before writing any code I worked out what the real constraint was, because 30 requests a minute isn't the only number Groq gives you. The error message that showed up on the first Groq run mentioned a second limit sitting underneath it:

```
Rate limit reached ... tokens per minute (TPM): Limit 8000, Used 7803, Requested 370.
```

So there are two ceilings, not one: 30 requests per minute, and 8000 tokens per minute. Requests per minute converts to one request every 2 seconds. But each of my requests (system prompt plus email plus the model's own reasoning tokens before it outputs JSON) was running 300 to 500 tokens, based on what I'd already seen logged from earlier test calls. At 400 tokens a request, the token budget only allows about 20 requests a minute, which works out to one request every 3 seconds. That's a tighter limit than the request-count cap, so it's the one that actually decides the pacing. Picking 3 seconds wasn't a guess, it came from dividing the tighter of the two published limits by my own measured token usage per call, with a little room left over since gpt-oss's reasoning length varies from one response to the next.

With that number in hand, the fix itself is a small file, `src/rate_limiter.py`:

```python
import asyncio
import time

MIN_INTERVAL_SECONDS = 3.0

_lock = asyncio.Lock()
_next_available_time = 0.0

async def throttle():
    global _next_available_time
    async with _lock:
        now = time.monotonic()
        wait_time = _next_available_time - now
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            now = time.monotonic()
        _next_available_time = now + MIN_INTERVAL_SECONDS
```

The idea is a single shared bookmark, `_next_available_time`, holding the earliest moment the next Groq request is allowed to go out. Every coroutine that wants to call Groq runs `await throttle()` first. Inside, it checks how far away that bookmark is. If it's in the future, the coroutine sleeps until then. Either way, before it lets go, it pushes the bookmark forward by another 3 seconds, so the next caller in line has to wait its turn too.

The `asyncio.Lock` is what makes this safe with multiple test cases running at once. Without it, two coroutines could both read `_next_available_time` at the same moment, both see it's fine to go, and both fire immediately, which defeats the whole point. The lock forces the check-and-update to happen one coroutine at a time, so the requests actually come out spaced apart rather than in clumps.

The other detail that mattered: this function lives in its own module and gets imported into both `classifier.py` and `evaluator.py`, rather than being copied into each. Both of those files talk to the same Groq account, so the 30 requests a minute limit is shared between them whether I account for that or not. If each file had its own copy of the throttle, each would think it had the full 30 requests a minute to itself, and the two together would still double up on Groq's real limit. One shared bookmark, imported everywhere it's needed, is what makes the pacing actually match the account-wide limit rather than an imagined per-file one.

I kept the `asyncio.Semaphore(5)` around in `evaluator.py` too, but its job changed. It used to be my attempt at rate limiting, which didn't work because it only bounds how many requests can be in the air at once, not how often new ones start. Now that the throttle handles the real pacing, the semaphore's only job is to stop all 51 test cases from queuing up on the lock at the exact same instant when the script starts, which is a minor scheduling nicety rather than the thing keeping me under Groq's limit.

## The same thing, in plain terms

The code above is dense if you don't read async Python often, so here's the same idea without the syntax.

**What a semaphore actually is.** Think of a parking lot with exactly 5 spots. Cars (test cases) show up whenever they want. If a spot is open, a car pulls in and starts. If all 5 spots are full, the next car just waits at the entrance until someone leaves. `asyncio.Semaphore(5)` is that parking lot: it lets up to 5 pieces of code run at the same time, and makes anything past that limit wait its turn. It caps how many things happen *at once*. It says nothing about how often a new one is allowed to start.

That's exactly why it didn't solve the Groq problem. A parking lot with 5 spots doesn't stop cars from arriving in quick bursts, it only stops more than 5 from being there simultaneously. Groq doesn't care how many requests I have "parked" at once, it cares how many happen within a rolling 60 second window. A 5-spot parking lot and a 30-cars-per-minute road are two completely different rules, and I'd only built the first one.

**What the throttle does differently.** Forget parking spots. Picture a single-lane tollbooth with a strict rule: exactly one car through every 3 seconds, no matter how many cars are waiting behind it. That's `throttle()`. It doesn't care how many test cases are running in the background. It only cares that whatever calls it next has to wait until 3 seconds have passed since the last car went through the booth.

The `_next_available_time` variable is just a sign at the tollbooth that says "next car may pass at 10:04:23." Every coroutine that wants to talk to Groq walks up, reads that sign, and either goes through immediately (if the time has already passed) or waits until it has. Right before it drives through, it updates the sign to say "next car may pass 3 seconds from now," so whoever's behind it has to wait too.

**Why it needs a lock at all.** Without the lock, two coroutines could both walk up to the sign at the exact same instant, both read "you're clear to go," and both drive through together, which is exactly the pile-up I was trying to prevent. The `asyncio.Lock` just means only one coroutine is allowed to read-and-update the sign at a time. Everyone else waits their turn to even *check* the sign, so two cars can never both think the coast is clear simultaneously.

Put together: the semaphore is the parking lot that stops too many test cases from being active at once (mostly for tidiness), and the throttle is the tollbooth that actually enforces "no more than one Groq request every 3 seconds" no matter what the parking lot is doing. They're solving two different problems that happen to sound similar, which is exactly why my first attempt (parking lot only) didn't work, and why the fix needed the tollbooth added on top rather than instead of it.

## Where it landed

```
Category Accuracy: 88.2% (45/51)
Avg Summary Score: 4.76 / 5.0
Total Tokens:      15,832
Total Wall Time:   303.78 seconds
```

## What I'd take from this

A free tier's limit shape matters more than the raw number attached to it. 50 requests a day sounds generous until you check it against your actual workload. 30 requests a minute sounds stingy until you realize it just means waiting a few minutes, not coming back tomorrow.

Don't keep fixing the symptom in front of you if there's a wall behind it. The overload handling I built for OpenRouter was fine work, it just didn't matter once the daily cap showed up underneath it. Worth checking whether something is actually fixable before spending more time on it.

Old commented out code in a repo isn't clutter, it's often a note from your past self. The dead Groq client line was a five second hint that someone had already scoped an alternative.

Ask before spending someone else's money. Paying to raise OpenRouter's cap would have worked, but reaching for a paid fix without checking first isn't the right default when a free path hasn't been ruled out yet.
