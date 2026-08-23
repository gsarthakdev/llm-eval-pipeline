from src.classifier import classify_email_async
from src.evaluator import score_summary_relevance
dummy_email = """

Hi support,
I Tried logging in today, but the system keeps throwing a "500 internal server error" every time I click the dashboard link. Because of this, I can't update my credit card before the renewal date tomorrow! Please help.
Thanks,
Sarah
"""

async def call_llm_as_judge():
    score = await score_summary_relevance("User wants to cancel their premium plan but is blocked by app crashes.", "The user is experiencing app crashes when trying to cancel their pro plan, leading to continued charges.")
    return score

if __name__ == "__main__":
    """
    print("Sending email to LLM...")
    result = classify_email_async(dummy_email, "prompts/support_v1.yaml")
    print("\n---- LLM Response ---- ")
    # print(f"Cateogory: {result.category}")
    # print(f"Summary: {result.summary}")
    print(result)
    print("------------------------\n")
    """
    # TC-051
    import asyncio
    result = asyncio.run(score_summary_relevance("User wants to cancel their premium plan but is blocked by app crashes.", "The user is experiencing app crashes when trying to cancel their pro plan, leading to continued charges."))
    print(result)
    # print(type(result.choices[0].message.parsed.relevance_score))
   
    
