import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

load_dotenv()

DB_FOLDER = "chroma_db"
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_collection(
    name="cooperative_docs",
    embedding_function=embedding_fn
)


def retrieve_relevant_chunks(question, n_results=5):
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    return results["documents"][0]


def generate_answer(question, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant for Indian cooperative society members.
Answer the user's question using ONLY the context below.
If the context doesn't contain the answer, say you don't have that information.

Only state specific numbers, timelines, or formulas if they appear EXACTLY in the context below. If the context doesn't contain a specific figure, timeline, or number, say "the provided documents don't specify this" rather than estimating or giving an example.

Context:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def rag_pipeline(question):
    chunks = retrieve_relevant_chunks(question)
    answer = generate_answer(question, chunks)
    return answer


if __name__ == "__main__":
    test_q = "What documents do I need to join a PACS?"
    print(f"Q: {test_q}")
    print(f"A: {rag_pipeline(test_q)}")