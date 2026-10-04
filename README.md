# LLM Eval Pipeline

A testing harness for AI prompts. It answers one question before a prompt change ships: did this get worse at its job?

## The problem

Say you have a customer support email classifier running on an LLM. Someone edits the system prompt to fix a bug in how it handles billing questions, and the edit quietly breaks how it handles account questions. Nobody notices until a customer complains. This pipeline catches that before the prompt ever reaches production.

## How it works

1. **Golden dataset.** 51 hand-labeled support emails in [data/golden_dataset.json](data/golden_dataset.json), each tagged with the correct category (`billing`, `technical`, `account`, `general`) and a reference summary.
2. **Classification.** Each email runs through the candidate prompt. The model returns a category and a one-sentence summary, validated against a Pydantic schema in [src/models.py](src/models.py).
3. **Scoring.** Category accuracy is exact match against the label. Summary quality uses a second LLM call as a judge, rating semantic similarity to the reference summary on a scale of 1 to 5.
4. **Diffing.** [src/differ.py](src/differ.py) and [src/reporter.py](src/reporter.py) compare the candidate run against a stored baseline, case by case. Any test that flips from passing to failing is flagged as a regression. A rolling average over the last 7 runs catches slow drift that a single run would miss.
5. **Reporting.** An HTML diff report lists every regression and improvement side by side. A Slack webhook can post the summary straight to a team channel.

All 51 cases run concurrently through `asyncio`, so a full evaluation finishes in under five minutes instead of one request at a time.

## Why the rate limiter exists

The classifier started on OpenAI. When those credits ran out, I moved it to Groq's free tier, which caps usage at 30 requests per minute and 8,000 tokens per minute. Firing 51 test cases at once blew past both limits in seconds.

[src/rate_limiter.py](src/rate_limiter.py) fixes that with a single shared timestamp: the earliest moment the next request is allowed to go out. Every call to Groq checks that timestamp, waits if it's in the future, then pushes it forward before releasing control. An `asyncio.Lock` stops two concurrent requests from reading the same timestamp at once and both slipping through together. One shared pace, enforced across every file that talks to Groq, instead of each one guessing its own limit.

The 3-second interval isn't arbitrary either. Groq's real bottleneck turned out to be the token budget, not the request count: 8,000 tokens per minute divided by the ~400 tokens a typical classification call uses works out to roughly one request every 3 seconds. The full writeup is in [notes/openrouter_to_groq_migration.md](notes/openrouter_to_groq_migration.md).

## Running it

```bash
pip install openai pydantic pyyaml python-dotenv requests
```

Set a `GROQ_API_KEY` in a `.env` file at the project root. A `SLACK_WEBHOOK_URL` is optional, for alerting on regressions.

Run a full evaluation against a prompt:

```bash
python3 -m src.evaluator
```

Compare two runs and generate a diff report:

```bash
python3 -m src.reporter
```

## Project layout

```
src/
  classifier.py   # calls the model, validates its output against a schema
  evaluator.py     # runs the golden dataset against a candidate prompt
  differ.py        # flags regressions and improvements between two runs
  reporter.py      # builds the HTML report, tracks drift, posts to Slack
  rate_limiter.py  # keeps every request under Groq's rate limits
  models.py        # Pydantic schemas for classifier and judge output
prompts/
  support_v1.yaml  # the system prompt under test
data/
  golden_dataset.json  # the 51 labeled test cases
  eval_runs/           # stored run snapshots and baselines
```

## What's next

Right now a run has to be triggered manually. The next step is wiring `src/main.py` into a CI job so every prompt change gets evaluated automatically against the baseline before merge.
