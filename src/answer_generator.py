from langchain_community.llms import Ollama
from query_analyzer import analyze_query
from retriever import retrieve
from context_builder import enrich

llm = Ollama(model="llama3")

def ask(query):
    analysis = analyze_query(query)

    results = retrieve(query, analysis)

    context = enrich(results)

    prompt = f"""
You are an STM32 expert.

Context:
{context}

Question:
{query}

Answer with technical precision.
"""

    return llm.invoke(prompt)
