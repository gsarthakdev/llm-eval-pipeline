from src.classifier import classify_email

dummy_email = """

Hi support,
I Tried logging in today, but the system keeps throwing a "500 internal server error" every time I click the dashboard link. Because of this, I can't update my credit card before the renewal date tomorrow! Please help.
Thanks,
Sarah
"""

if __name__ == "__main__":
    print("Sending email to LLM...")
    result = classify_email(dummy_email, "prompts/support_v1.yaml")
    print("\n---- LLM Response ---- ")
    # print(f"Cateogory: {result.category}")
    # print(f"Summary: {result.summary}")
    print(result)
    print("------------------------\n")
