from typing import Literal
from pydantic import BaseModel, Field

class EmailInput(BaseModel):
    email_text: str

class ClassificationOutput(BaseModel):
    # category: str = Field(description="Must be one of: billing, technical, account, general")
    category: Literal["billing", "technical", "account", "general"] = Field(description="The primary classification cateogry for the incoming customer email.")
    summary: str = Field(description="A concise one-sentence summary of the email.")
    