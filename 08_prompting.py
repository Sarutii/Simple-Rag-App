import os
from importlib import import_module


context_building = import_module("07_context_building")
store = import_module("05_create_vector_store")

MOCK_MODE = not os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        _collection = store.create_vector_store()
    return _collection


def build_strict_prompt(query, context_text):
    return f"""You are a grounded RAG assistant.

Rules:
1. Use only the provided context. Never add background knowledge.
2. If the answer is not in the context, say: "The provided sources do not contain enough information to answer this question."
3. If a source is marked OUTDATED, do not use it as the primary answer. Mention it only to note the conflict.
4. If current and outdated sources conflict, state the conflict and use the CURRENT source.
5. Output exactly two sections:
   Answer: [your grounded answer]
   Sources: [list the source numbers you used]

Question:
{query}

Context:
{context_text}
"""


def ask_llm(prompt):
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def answer_question(question):
    collection = get_collection()
    package = context_building.build_context_package(collection, question)
    prompt = build_strict_prompt(question, package["context_text"])
    sources = package["selected_df"].to_dict("records")
    return ask_llm(prompt), sources


if __name__ == "__main__":
    answer, sources = answer_question("How much does an official transcript cost right now?")
    print(answer)
    print("\nsource doc ids:", [s["document_id"] for s in sources])
