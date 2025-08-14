import argparse
import configparser
import json
import logging
import os
import re
import sqlite3
import time
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import faiss
import httpx
import numpy as np
import tiktoken
import torch
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from libzim import Archive
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- Google API Imports ---
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- Application Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
log = logging.getLogger('oracle-server')
RAG_PROMPT_HEADER = "You are an expert research assistant."

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False

# --- Core Functions ---
def load_config(config_path: str):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def embed_texts(texts: List[str], model) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True)

def plain_text(html: bytes) -> str:
    soup = BeautifulSoup(html, 'lxml')
    for t in soup(['script', 'style', 'table', 'nav', 'footer', 'header', 'aside']):
        t.decompose()
    return re.split(r'\bSee also\b|\bReferences\b', soup.get_text(' ', strip=True), maxsplit=1)[0]

def trim_text(text: str, n_tok: int, tokenizer) -> str:
    return tokenizer.decode(tokenizer.encode(text)[:n_tok])

_html_cache: dict[str, str] = {}
def fetch_html(path: str, archives) -> str:
    if path in _html_cache: return _html_cache[path]
    try:
        zim_stem, article_path = path.split('/', 1)
        html = bytes(archives[zim_stem].get_entry_by_path(article_path).get_item().content).decode('utf-8', 'ignore')
        if len(_html_cache) > 200: _html_cache.pop(next(iter(_html_cache)))
        _html_cache[path] = html
        return html
    except Exception as e:
        log.warning(f"Could not fetch HTML for path '{path}': {e}")
        return ""

class Retriever:
    def __init__(self, faiss_path, db_path, top_k):
        log.info(f"Loading FAISS index from {faiss_path}...")
        self.index = faiss.read_index(str(faiss_path))
        log.info(f"Connecting to metadata db at {db_path}...")
        self.db_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.top_k = top_k

    def search(self, query: str, request: Request) -> List[Dict]:
        q_vec = embed_texts([query], request.app.state.embedding_model)
        scores, ids = self.index.search(q_vec, self.top_k)
        results = []
        for s, i in zip(scores[0], ids[0]):
            if i == -1: continue
            row = self.db_conn.execute('SELECT path, title FROM meta WHERE id=?', (int(i) + 1,)).fetchone()
            if row:
                results.append(dict(zip(('score', 'path', 'title'), (float(s),) + row)))
        return results

    def close(self):
        self.db_conn.close()

def build_rag_prompt(query: str, request: Request) -> str:
    config = request.app.state.config
    retriever = request.app.state.retriever
    tokenizer = request.app.state.tokenizer
    archives = request.app.state.zim_archives
    token_budget = config.getint('RAG', 'token_budget')
    
    docs = retriever.search(query, request)
    if not docs:
        log.warning(f"No documents found for query: '{query}'")
        return ""

    retrieved_titles = [d['title'] for d in docs]
    log.info(f"Retrieved context for query '{query}': {retrieved_titles}")

    contexts = [f"Article: {d['title']}\n{trim_text(plain_text(fetch_html(d['path'], archives).encode('utf-8')), 512, tokenizer)}" for d in docs]
    full_context = trim_text("\n\n---\n\n".join(contexts), token_budget, tokenizer)
    
    system_prompt = (
        f"{RAG_PROMPT_HEADER} Your task is to provide a detailed, factual answer to the user's question using ONLY the provided text excerpts. You must follow these rules without exception:\n\n"
        "1.  **Synthesize, Don't Invent:** Your answer must be entirely derived from the information within the 'Provided Excerpts' section.\n"
        "2.  **Cite Your Sources:** After every statement or paragraph, include a citation `[Source: Title]` referencing the article the information came from.\n"
        "3.  **Handle Missing Information:** If the excerpts do not contain the information to answer the question, you must reply with the exact phrase: `The provided knowledge base does not contain sufficient information to answer this question.`\n"
        "4.  **No Disclaimers:** Do not include apologies or disclaimers.\n\n"
        f"## Provided Excerpts:\n---\n{full_context}"
    )
    return system_prompt

# --- API Backend Handlers ---
async def handle_openai_compatible(req: ChatRequest, final_messages: List[Dict], request: Request) -> JSONResponse | StreamingResponse:
    client = request.app.state.llm_client
    prefix = request.app.state.config.get('RAG', 'model_prefix')
    timeout = request.app.state.config.getfloat('Server', 'request_timeout')
    backend_model_name = req.model.replace(prefix, "", 1)
    
    final_payload = req.model_dump()
    final_payload['model'] = backend_model_name
    final_payload["messages"] = final_messages

    endpoint = "/v1/chat/completions"
    
    if req.stream:
        async def stream_generator():
            async with client.stream("POST", endpoint, json=final_payload, timeout=timeout) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes(): yield chunk
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        resp = await client.post(endpoint, json=final_payload, timeout=timeout)
        resp.raise_for_status()
        return JSONResponse(content=resp.json())

async def handle_openai(req: ChatRequest, final_messages: List[Dict], request: Request) -> JSONResponse | StreamingResponse:
    client = request.app.state.llm_client
    virtual_model_name = req.model
    backend_model_name = request.app.state.config.get('openai', 'model')
    
    if req.stream:
        stream = await client.chat.completions.create(model=backend_model_name, messages=final_messages, stream=True)
        async def stream_generator():
            async for chunk in stream:
                chunk_dict = chunk.model_dump()
                chunk_dict['model'] = virtual_model_name
                yield f"data: {json.dumps(chunk_dict)}\n\n"
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        completion = await client.chat.completions.create(model=backend_model_name, messages=final_messages, stream=False)
        response_data = completion.model_dump()
        response_data['model'] = virtual_model_name
        return JSONResponse(content=response_data)

async def handle_anthropic(req: ChatRequest, final_messages: List[Dict], request: Request) -> JSONResponse | StreamingResponse:
    client = request.app.state.llm_client
    virtual_model_name = req.model
    backend_model_name = request.app.state.config.get('anthropic', 'model')
    
    system_prompt = next((m['content'] for m in final_messages if m['role'] == 'system'), "")
    user_messages = [m for m in final_messages if m['role'] != 'system']
    
    if req.stream:
        async def stream_generator():
            async with client.messages.stream(
                model=backend_model_name,
                system=system_prompt,
                messages=user_messages,
                max_tokens=4096
            ) as stream:
                async for chunk in stream.text_stream:
                    mock_chunk = {"id": f"chatcmpl-anthropic-{os.urandom(4).hex()}", "object": "chat.completion.chunk", "created": int(time.time()), "model": virtual_model_name, "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]}
                    yield f"data: {json.dumps(mock_chunk)}\n\n"
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        completion = await client.messages.create(model=backend_model_name, system=system_prompt, messages=user_messages, max_tokens=4096)
        response_data = {"id": completion.id, "model": virtual_model_name, "object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": completion.content[0].text}}]}
        return JSONResponse(content=response_data)

async def handle_google(req: ChatRequest, final_messages: List[Dict], request: Request) -> JSONResponse | StreamingResponse:
    client = request.app.state.llm_client
    virtual_model_name = req.model
    
    system_prompt = next((m['content'] for m in final_messages if m['role'] == 'system'), "")
    user_messages = [m for m in final_messages if m['role'] != 'system']
    
    gemini_messages = [{"role": "user", "parts": [f"{system_prompt}\n\nUSER QUESTION: {user_messages[0]['content']}"]}]
    gemini_messages.extend([{"role": "model" if m["role"] == "assistant" else "user", "parts": [m["content"]]} for m in user_messages[1:]])
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    if req.stream:
        async def stream_generator():
            try:
                response_iterator = await client.generate_content_async(contents=gemini_messages, stream=True, safety_settings=safety_settings)
                async for chunk in response_iterator:
                    if chunk.parts:
                        mock_chunk = {"id": f"chatcmpl-google-{os.urandom(4).hex()}", "object": "chat.completion.chunk", "created": int(time.time()), "model": virtual_model_name, "choices": [{"index": 0, "delta": {"content": chunk.text}, "finish_reason": None}]}
                        yield f"data: {json.dumps(mock_chunk)}\n\n"
            except StopAsyncIteration: pass
            except Exception as e: log.error(f"An error occurred during Google stream generation: {e}")
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        completion = await client.generate_content_async(contents=gemini_messages, safety_settings=safety_settings)
        completion_text = completion.text if getattr(completion, 'parts', None) else "The model did not provide a response. This may be due to the provider's safety filters."
        response_data = {"id": f"chatcmpl-google-{os.urandom(4).hex()}", "model": virtual_model_name, "object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": completion_text}}]}
        return JSONResponse(content=response_data)


PROVIDER_HANDLERS = {
    "openai-compatible": handle_openai_compatible,
    "ollama": handle_openai_compatible,
    "openai": handle_openai,
    "anthropic": handle_anthropic,
    "google": handle_google,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config(app.state.config_path)
    app.state.config = config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Using device: {device} for embeddings.")
    app.state.embedding_model = SentenceTransformer(config.get('Builder', 'embedding_model'), device=device)
    app.state.tokenizer = tiktoken.get_encoding('cl100k_base')
    faiss_path = Path(config.get('Paths', 'output_path')).with_suffix('.faiss')
    meta_db_path = Path(config.get('Paths', 'output_path')).with_suffix('.sqlite')
    if faiss_path.exists() and meta_db_path.exists():
        app.state.retriever = Retriever(faiss_path, meta_db_path, config.getint('RAG', 'top_k'))
    else:
        app.state.retriever = None
        log.error("---!!! KNOWLEDGE BASE NOT FOUND !!!---")
    app.state.zim_archives = {Path(p.strip()).stem: Archive(str(Path(p.strip()))) for p in config.get('Paths', 'zim_files').split(',')}
    provider = config.get('LLM', 'provider').lower()
    if provider in ['ollama', 'openai-compatible']:
        base_url = config.get(provider, 'backend_url')
        app.state.llm_client = httpx.AsyncClient(base_url=base_url)
    elif provider == 'openai':
        from openai import AsyncOpenAI
        api_key = os.getenv("OPENAI_API_KEY") or config.get('openai', 'api_key')
        app.state.llm_client = AsyncOpenAI(api_key=api_key)
    elif provider == 'anthropic':
        from anthropic import AsyncAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY") or config.get('anthropic', 'api_key')
        app.state.llm_client = AsyncAnthropic(api_key=api_key)
    elif provider == 'google':
        api_key = os.getenv("GOOGLE_API_KEY") or config.get('google', 'api_key')
        genai.configure(api_key=api_key)
        model_name = config.get('google', 'model')
        app.state.llm_client = genai.GenerativeModel(model_name)
    log.info(f"--- Offline Oracle server ready. Provider: '{provider}' ---")
    yield
    if app.state.retriever: app.state.retriever.close()
    if hasattr(app.state, 'llm_client') and isinstance(app.state.llm_client, httpx.AsyncClient): await app.state.llm_client.aclose()
    log.info("--- Server shutdown complete ---")

app = FastAPI(lifespan=lifespan, title="Offline Oracle RAG Server")

@app.get("/v1/models")
async def list_models(request: Request):
    config = request.app.state.config
    provider = config.get('LLM', 'provider').lower()
    prefix = config.get('RAG', 'model_prefix')
    if provider in ['ollama', 'openai-compatible']:
        client = request.app.state.llm_client
        try:
            endpoint = "/v1/models" if provider == 'openai-compatible' else "/api/tags"
            model_key = "data" if provider == 'openai-compatible' else "models"
            name_key = "id" if provider == 'openai-compatible' else "name"
            backend_response = await client.get(endpoint)
            backend_response.raise_for_status()
            backend_data = backend_response.json()
            model_list = [m[name_key] for m in backend_data.get(model_key, [])]
            modified_models = [{"id": f"{prefix}{name}", "object": "model"} for name in model_list if "embed" not in name.lower()]
            return {"object": "list", "data": modified_models}
        except Exception as e:
            log.error(f"Could not fetch models from '{provider}' backend: {e}")
            raise HTTPException(status_code=500, detail=f"Could not connect to backend at {client.base_url}")
    else:
        virtual_model_id = f"{prefix}{provider.capitalize()}"
        return {"object": "list", "data": [{"id": virtual_model_id, "object": "model"}]}

@app.post("/v1/chat/completions")
async def completions(req: ChatRequest, request: Request):
    if not request.app.state.retriever:
        raise HTTPException(status_code=503, detail="Knowledge Base not built or loaded.")
    
    # In this robust model, every user query that is part of the RAG interaction
    # will trigger a search. This ensures maximum verifiability.
    # The 'query' for the search is always the most recent user message.
    query = next((m["content"] for m in reversed(req.messages) if m["role"] == "user"), None)
    if not query:
        raise HTTPException(status_code=400, detail="No user query found in the message history.")
    
    log.info(f"Performing RAG search for the latest query.")
    system_prompt = build_rag_prompt(query, request)
    if not system_prompt:
        return JSONResponse(content={"choices": [{"message": {"role": "assistant", "content": "The provided knowledge base does not contain sufficient information to answer this question."}}]})
    
    # Construct the final message list for the LLM
    final_messages = [{"role": "system", "content": system_prompt}]
    # Add the entire user-assistant history, filtering out any pre-existing system prompts from the client.
    final_messages.extend([m for m in req.messages if m['role'] != 'system'])

    provider = request.app.state.config.get('LLM', 'provider').lower()
    handler = PROVIDER_HANDLERS.get(provider)
    if not handler:
        raise HTTPException(status_code=501, detail=f"Provider '{provider}' is not implemented.")
        
    return await handler(req, final_messages, request)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs the Offline Oracle RAG server.")
    parser.add_argument('--config', default='config.ini', help='Path to the configuration file.')
    args = parser.parse_args()
    app.state.config_path = args.config
    config = load_config(args.config)
    uvicorn.run(app, host=config.get('Server', 'host'), port=config.getint('Server', 'port'), reload=False)