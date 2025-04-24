# server.py
import os
import asyncio
import re
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import GoogleSerperAPIWrapper
from duckduckgo_search import DDGS
from transformers import pipeline

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Make sure you've run:
#   ollama serve tinyllama:chat
# before starting this FastAPI app.

# Initialize services
llm = OllamaLLM(model="tinyllama:chat")
prompt = PromptTemplate(input_variables=["query"], template="User: {query}\nAssistant:")
llm_chain = LLMChain(llm=llm, prompt=prompt)
search_tool = GoogleSerperAPIWrapper()
summarizer = pipeline("summarization", model="t5-small", device=-1)

app = FastAPI(title="Streaming Hybrid Chatbot")

# ─── UTILITY FUNCTIONS ────────────────────────────────────────────────────────

async def stream_text(text: str):
    """Stream text chunk by chunk."""
    for chunk in text:
        yield chunk
        await asyncio.sleep(0)

async def stream_llm(prompt_text: str, anti_hallucination=True):
    """Stream LLM responses with anti-hallucination protection."""
    if anti_hallucination:
        # Add clear instructions to reduce hallucination
        prompt_text = (
            "Answer the following question based ONLY on information you're certain about. "
            "If you don't know or are unsure, clearly state that you don't have enough information. "
            "Do NOT use phrases like 'I believe', 'probably', or 'might be'. Be direct and factual.\n\n"
            f"Question: {prompt_text}"
        )
        
        # Add disclaimer
        yield "Note: This response is based on my training data and may not reflect current information.\n\n"
    
    # Stream the actual response
    try:
        async for token in llm.astream(prompt_text):
            yield token
            await asyncio.sleep(0)
    except Exception as e:
        yield f"Error generating response: {str(e)}"

async def get_web_results(query):
    """Get web search results with fallback mechanism. Returns enhanced results."""
    snippets = []
    links = []
    titles = []
    
    # Try Serper first
    try:
        docs = search_tool.results(query).get("organic", [])
        for doc in docs[:5]:
            snippets.append(doc["snippet"])
            links.append(doc["link"])
            titles.append(doc.get("title", "No title"))
    except Exception:
        pass
    
    # Fallback to DuckDuckGo
    if not snippets:
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=5)]
                for result in results:
                    snippets.append(result["body"])
                    links.append(result["link"])
                    titles.append(result.get("title", "No title"))
        except Exception:
            pass
    
    return {
        "snippets": snippets,
        "links": links,
        "titles": titles
    }

async def summarize_snippets(snippets):
    """Summarize web snippets."""
    if not snippets:
        return ""
    
    try:
        combined = " ".join(snippets)
        summary = summarizer(combined, max_length=150, min_length=50)[0]["summary_text"]
        return summary
    except Exception:
        # If summarization fails, return the first snippet
        return snippets[0] if snippets else ""

def create_web_prompt(query, web_context):
    """Create a prompt for LLM using web context."""
    return (
        f"[WEB CONTEXT]\n{web_context}\n\n"
        f"[QUERY]\n{query}\n\n"
        "Instructions:\n"
        "1. Answer ONLY based on information in the web context above\n"
        "2. If the web context doesn't have enough information, clearly say so\n"
        "3. Do NOT add information beyond what's provided\n"
        "4. Be direct and factual\n"
        "5. If you're unsure, say 'Based on the web information, I can't answer this completely'"
    )

def assess_llm_response(response):
    """Assess if an LLM response is adequate."""
    response = response.lower()
    
    # Check for uncertainty markers
    uncertainty_phrases = [
        "i don't know", "as a language model", "i cannot", "i'm not sure",
        "i don't have", "i don't have access", "i cannot access", 
        "i don't have information", "i don't have current", "i don't have up-to-date",
        "it depends", "it varies", "i can't provide"
    ]
    
    # Check if response contains uncertainty phrases
    if any(phrase in response for phrase in uncertainty_phrases):
        return False, "The LLM expressed uncertainty or knowledge limitations"
    
    # Check if response is too short
    if len(response) < 50 or len(response.split()) < 10:
        return False, "The LLM response was too brief"
        
    # Check if response has enough sentences
    if response.count(".") < 2:
        return False, "The LLM response lacked complete sentences"
    
    return True, "The LLM provided a complete response"

def needs_web_search(query, llm_response):
    """Determine if web search is necessary based on user query and LLM response."""
    query = query.lower()
    
    # 1. Check for explicit user requests for internet info
    web_request_terms = [
        "recent", "latest", "current", "today", "news", "update",
        "search", "look up", "find", "internet", "web", "online",
        "fact", "data", "statistics", "report", "2023", "2024", "2025"
    ]
    
    if any(term in query for term in web_request_terms):
        return True, "User explicitly requested current/factual information"
    
    # 2. Check questions about current events, recent developments
    current_event_patterns = [
        r"what happened (in|with|to|during)",
        r"how is .+ (doing|performing|going)",
        r"current state of",
        r"latest (news|development|update|version)",
        r"what is the status of"
    ]
    
    if any(re.search(pattern, query) for pattern in current_event_patterns):
        return True, "Query refers to current events or recent developments"
    
    # 3. Check for fact-based questions
    fact_patterns = [
        r"how many", r"when (did|was|will)", r"where is", 
        r"who is", r"what is the (price|cost|value) of"
    ]
    
    if any(re.search(pattern, query) for pattern in fact_patterns):
        return True, "Query is fact-based, best augmented with web data"
    
    # 4. Check if LLM response is inadequate
    is_adequate, _ = assess_llm_response(llm_response)
    if not is_adequate:
        return True, "LLM response was inadequate or uncertain"
    
    # Default: No need for web search
    return False, "LLM response is adequate and query doesn't require recent information"

# ─── API ENDPOINTS ──────────────────────────────────────────────────────────

@app.post("/chat/llm_only")
async def chat_llm_only(req: Request):
    """LLM-only endpoint with anti-hallucination measures."""
    try:
        data = await req.json()
        query = data.get("prompt")
        
        if not query:
            raise HTTPException(400, "Missing 'prompt'")
            
        return StreamingResponse(
            stream_llm(query, anti_hallucination=True),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(500, f"Error processing request: {str(e)}")

@app.get("/search")
async def search_only(q: str):
    """Search-only endpoint with enhanced results."""
    try:
        # Get search results
        results = await get_web_results(q)
        snippets = results["snippets"]
        links = results["links"]
        titles = results["titles"]
        
        if not snippets:
            raise HTTPException(404, f"No search results found for '{q}'")
        
        # Summarize
        summary = await summarize_snippets(snippets)
        
        # Format response with sources
        response = f"📝 Summary: {summary}\n\n📚 Sources:\n"
        for i in range(min(len(links), len(titles))):
            response += f"{i+1}. {titles[i]}\n   {links[i]}\n\n"
        
        return StreamingResponse(stream_text(response), media_type="text/plain")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error searching: {str(e)}")

@app.post("/chat/hybrid")
async def chat_hybrid(req: Request):
    """Hybrid endpoint that intelligently combines LLM with web search when needed."""
    try:
        data = await req.json()
        query = data.get("prompt")
        
        if not query:
            raise HTTPException(400, "Missing 'prompt'")

        # 1. Get LLM response first
        llm_response = ""
        async for token in llm.astream(query):
            llm_response += token
        
        # 2. Assess if web search is needed
        need_web, reason = needs_web_search(query, llm_response)

        # 3. If web search is NOT needed, just return the LLM response
        if not need_web:
            async def stream_llm_only():
                yield "🤖 [LLM Response]\n\n"
                yield llm_response
                yield "\n\n📌 Note: This response is based on the model's training data."
            
            return StreamingResponse(stream_llm_only(), media_type="text/plain")
        
        # 4. If web search IS needed, get web results
        web_results = await get_web_results(query)
        snippets = web_results["snippets"]
        links = web_results["links"]
        titles = web_results["titles"]
        
        # 5. If we have web results, augment the response
        if snippets:
            # Summarize web results
            web_summary = await summarize_snippets(snippets)
            
            # Create web-informed prompt
            web_prompt = create_web_prompt(query, web_summary)
            
            # Generate augmented response
            augmented_response = ""
            async for token in llm.astream(web_prompt):
                augmented_response += token
            
            # Stream the comprehensive response
            async def stream_augmented_response():
                # Original LLM response
                yield "🤖 [Original LLM Response]\n\n"
                yield llm_response
                
                # Reason for augmentation
                yield f"\n\n⚠️ Reason for web augmentation: {reason}\n\n"
                
                # Web summary
                yield f"🌐 [Web Summary]\n{web_summary}\n\n"
                
                # Augmented response
                yield "🔄 [Web-Augmented Response]\n\n"
                yield augmented_response
                
                # Sources
                yield "\n\n📚 Sources:\n"
                for i in range(min(len(links), len(titles))):
                    yield f"{i+1}. {titles[i]}\n   {links[i]}\n\n"
            
            return StreamingResponse(stream_augmented_response(), media_type="text/plain")
        
        # 6. If no web results found, explain and return LLM response
        async def stream_fallback():
            yield "🤖 [LLM Response]\n\n"
            yield llm_response
            
            yield f"\n\n⚠️ Note: Web search was attempted because: {reason}"
            yield "\nHowever, no relevant web results were found."
            yield "\nThis response is based solely on the model's training data."
        
        return StreamingResponse(stream_fallback(), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(500, f"Error processing hybrid request: {str(e)}")

# ─── TESTING FUNCTIONS ───────────────────────────────────────────────────────

async def test_llm_response(query):
    """Test LLM responses directly."""
    result = ""
    async for token in stream_llm(query):
        result += token
    return result

async def test_search(query):
    """Test search functionality directly."""
    results = await get_web_results(query)
    summary = await summarize_snippets(results["snippets"])
    return {
        "summary": summary,
        "snippets": results["snippets"][:2],  # First two snippets
        "links": results["links"],
        "titles": results["titles"]
    }

async def test_hybrid(query):
    """Test hybrid functionality directly."""
    # Get LLM response
    llm_response = ""
    async for token in llm.astream(query):
        llm_response += token
    
    # Assess if web search is needed
    need_web, reason = needs_web_search(query, llm_response)
    
    if not need_web:
        return {
            "type": "llm-only",
            "llm_response": llm_response,
            "is_adequate": True,
            "reason": reason
        }
    
    # Get web results
    results = await get_web_results(query)
    snippets = results["snippets"]
    
    if snippets:
        # Get web summary
        web_summary = await summarize_snippets(snippets)
        
        # Create web prompt
        web_prompt = create_web_prompt(query, web_summary)
        
        # Get augmented response
        augmented_response = await llm_chain.apredict(query=web_prompt)
        
        return {
            "type": "web-augmented",
            "llm_response": llm_response,
            "is_adequate": False,
            "reason": reason,
            "web_summary": web_summary,
            "augmented_response": augmented_response,
            "sources": [{"title": t, "link": l} for t, l in zip(results["titles"], results["links"])]
        }
    else:
        return {
            "type": "llm-only",
            "llm_response": llm_response,
            "is_adequate": False,
            "reason": reason,
            "warning": "No web data found"
        }

# For direct testing from command line
if __name__ == "__main__":
    import sys
    import asyncio
    import json
    
    async def run_test():
        if len(sys.argv) < 3:
            print("Usage: python main.py [llm|search|hybrid] \"your query\"")
            return
        
        mode = sys.argv[1]
        query = sys.argv[2]
        
        if mode == "llm":
            result = await test_llm_response(query)
            print(result)
        elif mode == "search":
            result = await test_search(query)
            print(f"Summary: {result['summary']}\n\nSources:")
            for i, (title, link) in enumerate(zip(result["titles"], result["links"]), 1):
                print(f"{i}. {title}\n   {link}")
        elif mode == "hybrid":
            result = await test_hybrid(query)
            print(json.dumps(result, indent=2))
        else:
            print("Invalid mode. Use 'llm', 'search', or 'hybrid'")
    
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(run_test())



