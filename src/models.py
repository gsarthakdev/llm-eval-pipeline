from typing import Literal
from pydantic import BaseModel, Field

class EmailInput(BaseModel):
    email_text: str

class ClassificationOutput(BaseModel):
    # category: str = Field(description="Must be one of: billing, technical, account, general")
    category: Literal["billing", "technical", "account", "general"] = Field(description="The primary classification cateogry for the incoming customer email.")
    summary: str = Field(description="A concise one-sentence summary of the email.")

class ScoredSummaryRelevance(BaseModel):
    relevance_score: Literal[1,2,3,4,5] = Field(description="Semantic similarity and factual accuracy of the scored summary on a scale of 1 to 5.")
    