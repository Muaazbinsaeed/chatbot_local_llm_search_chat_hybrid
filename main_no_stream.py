
# server.py
import os
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import GoogleSerperAPIWrapper
from duckduckgo_search import DDGS
from transformers import pipeline

# ──── Configuration ────────────────────────────────────────────────────────────
# Make sure: export SERPER_API_KEY="your_serper_key" in your shell
# And `ollama serve tinyllama:chat` is running.

# 1️⃣ Initialize LLM + prompt chain
llm = OllamaLLM(model="tinyllama:chat")
prompt = PromptTemplate( input_variables=["query"], template="User: {query}\nAssistant:")
llm_chain = LLMChain(llm=llm, prompt=prompt)

# 2️⃣ Initialize search tool
search_tool = GoogleSerperAPIWrapper()

# 3️⃣ Light summarizer
summarizer = pipeline("summarization", model="t5-small", device=-1)

app = FastAPI(title="Hybrid Chatbot")

def needs_search(llm_text: str, user_query: str) -> bool:
    """Basic rules to decide if we should fetch live web data."""
    # a) explicit user signals
    for kw in ("recent", "latest", "current", "today", "news"):
        if kw in user_query.lower():
            return True
    # b) LLM low-confidence heuristics
    if len(llm_text) < 50:
        return True
    for phrase in ("i don't know", "as a language model"):
        if phrase in llm_text.lower():
            return True
    return False


# ──── Mode 1: LLM only ─────────────────────────────────────────────────────────
@app.post("/chat/llm_only")
async def chat_llm_only(req: Request):
    data = await req.json()
    q = data.get("prompt")
    if not q:
        raise HTTPException(400, "Missing ‘prompt’")
    resp = llm_chain.predict(query=q)
    return {"response": resp}


# ──── Mode 2: Search only ──────────────────────────────────────────────────────
@app.get("/search")
async def search_only(q: str):
    snippets = []
    
    # 1. Try Serper.dev with error handling
    try:
        docs = search_tool.results(q).get("organic", [])
        snippets = [d["snippet"] for d in docs[:5]]
    except Exception as e:
        print(f"Serper API error: {str(e)}")
        # Continue to fallback
    
    # 2. Fallback to DuckDuckGo if Serper failed or returned no results
    if not snippets:
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(q, max_results=5)]
                snippets = [r["body"] for r in results]
        except Exception as e:
            print(f"DuckDuckGo error: {str(e)}")
            return {"error": "Search services unavailable", "query": q}

    if not snippets:
        return {"error": "No results found", "query": q}

    # 3. Summarize results
    try:
        combined = " ".join(snippets)
        summary = summarizer(combined, max_length=100, min_length=30)[0]["summary_text"]
        return {"query": q, "summary": summary}
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        # Return raw snippets if summarization fails
        return {"query": q, "summary": snippets[0] if snippets else "No summary available"}


# ──── Mode 3: Intelligent Search + LLM ─────────────────────────────────────────
@app.post("/chat/hybrid")
async def chat_hybrid(req: Request):
    data = await req.json()
    q = data.get("prompt")
    if not q:
        raise HTTPException(400, "Missing ‘prompt’")

    # 1️⃣ First pass: plain LLM
    llm_resp = llm_chain.predict(query=q)

    # 2️⃣ Decide if we need live web data
    if needs_search(llm_resp, q):
        # fetch top snippets
        docs = search_tool.results(q).get("organic", [])
        snippets = [d["snippet"] for d in docs[:5]]
        if not snippets:
            snippets = [r["body"] for r in DDGS().text(q, max_results=5)]

        # summarize web snippets
        combined = " ".join(snippets)
        summary = summarizer(combined, max_length=100, min_length=30)[0]["summary_text"]

        # re-prompt LLM with web context
        hybrid_prompt = (
            f"[WEB SUMMARY]\n{summary}\n\n"
            f"[ORIGINAL QUESTION]\n{q}"
        )
        llm_resp = llm_chain.predict(query=hybrid_prompt)

    return {"response": llm_resp}



# curl -X POST http://localhost:8000/chat/llm_only \
#      -H "Content-Type: application/json" \
#      -d '{"prompt":"Explain quantum computing in simple terms"}'


# curl "http://localhost:8000/search?q=Latest%20AI%20news"


# curl -X POST http://localhost:8000/chat/hybrid \
#      -H "Content-Type: application/json" \
#      -d '{"prompt":"What are the latest breakthroughs in battery technology?"}'
