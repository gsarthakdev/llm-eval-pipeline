import asyncio
import os
import time
import yaml
from dotenv import load_dotenv
# from openai import OpenAI
from openai import AsyncOpenAI, APIStatusError
from src.models import ClassificationOutput
from src.rate_limiter import throttle

load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = AsyncOpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

MAX_RETRIES = 5

def load_prompt_config(filepath: str) -> dict:
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

async def classify_email_async(email_text: str, prompt_filepath: str) -> dict:
    config = load_prompt_config(prompt_filepath)
    start_time = time.time()

    for attempt in range(MAX_RETRIES):
        await throttle()
        try:
            response = await client.beta.chat.completions.parse(
                model=config["model"],
                messages=[
                    {"role": "system", "content": config["system_prompt"]},
                    {"role": "user", "content": email_text}
                ],
                response_format=ClassificationOutput
            )
        except APIStatusError as e:
            if attempt == MAX_RETRIES - 1 or e.status_code not in (429, 500, 502, 503):
                raise
            await asyncio.sleep(2 ** attempt)
            continue
        break

    # latency in seconds
    latency = time.time() - start_time

    return {
        "output": response.choices[0].message.parsed,
        "total_tokens": response.usage.total_tokens,
        "latency_seconds": latency
    }
