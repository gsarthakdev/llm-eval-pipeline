import os
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from src.models import ClassificationOutput

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_prompt_config(filepath: str) -> dict:
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)
    
def classify_email(email_text: str, prompt_filepath: str) -> ClassificationOutput:
    config = load_prompt_config(prompt_filepath)
    
    response = client.beta.chat.completions.parse(
        model=config["model"],
        messages=[
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": email_text}
        ],
        response_format=ClassificationOutput
    )
    
    return response.choices[0].message.parsed

