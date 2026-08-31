import time
from rag_pipeline import rag_pipeline

test_questions = [
    "What is the Multi-State Cooperative Societies Act about?",
    "What services does a PACS provide to farmers?",
    "What is the PACS computerization scheme?",
    "What is the claim settlement procedure under PMFBY?",
    "What crops are covered under RWBCIS?",
    "How can I identify and avoid a loan scam?",
    "What is PMJDY?",
    "What is the grievance redressal mechanism for cooperative societies?",
    "How can I file a complaint against my cooperative?",
]

for q in test_questions:
    print(f"Q: {q}")
    print(f"A: {rag_pipeline(q)}")
    print("-" * 80)
    time.sleep(12)  # avoid Groq rate limits