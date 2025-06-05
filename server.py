from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import os, json, tiktoken, openai, chromadb, time
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from uuid import uuid4

openai.api_key = os.getenv("OPENAI_API_KEY")
enc = tiktoken.encoding_for_model("gpt-4o-mini")

client = chromadb.PersistentClient(path="/data/chroma_store")  # <─ mount point on Render
def get_collection(keyword):
    return client.get_or_create_collection(
        name=f"user_{keyword}",
        embedding_function=OpenAIEmbeddingFunction(
            api_key=openai.api_key,
            model_name="text-embedding-3-small",
        )
    )

ROLE_BONUS = {"profile": 0.2, "user": 0.1, "assistant": 0}
SYS_TEMPLATE = """
You are **Hello Me**, an empathetic AI guide who helps users talk through issues
and recommends tiny next steps.

Conversation flow
1. Greet → “Hi, what’s on your mind today?”
2. Explore → ask clarifying questions.
3. Suggest a small action.
4. Close → “Let’s check in next time to see how it went.”

Below are past memories (use if relevant):
{snips}

Now respond in ≤2 sentences.
""".strip()

def recall_memories(keyword, query, k=4):
    col = get_collection(keyword)
    if col.count() == 0:
        return []
    res = col.query(query_texts=[query], n_results=12,
                    include=["documents", "metadatas", "distances"])
    scored = []
    for doc, meta, dist in zip(*res.values()):
        sim = 1 - dist
        bonus = ROLE_BONUS.get(meta["role"], 0)
        scored.append((sim + bonus, doc))
    scored.sort(reverse=True)
    return [d for _, d in scored[:k]]

def add_memory(keyword, role, text):
    col = get_collection(keyword)
    col.add(ids=[str(uuid4())],
            documents=[text],
            metadatas=[{"role": role, "ts": time.time()}])

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.post("/v1/chat/completions")
async def completions(req: Request):
    body = await req.json()
    messages = body["messages"]
    keyword  = (body.get("metadata") or {}).get("keyword", "anon")

    user_msg = messages[-1]["content"]
    snips    = recall_memories(keyword, user_msg)
    memo     = {"role": "system",
                "content": SYS_TEMPLATE.format(snips="\n".join(f"- {s}" for s in snips))}
    prompt   = [memo] + messages

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        max_tokens=200,
        temperature=0.7,
        stream=True,
    )

    async def streamer():
        full = []
        for chunk in resp:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", "") or ""
            full.append(content)
            if content:
                yield f'data: {json.dumps({"choices":[{"delta":{"content":content}}]})}\n\n'
        yield "data: [DONE]\n\n"
        add_memory(keyword, "user", user_msg)
        add_memory(keyword, "assistant", "".join(full))

    return StreamingResponse(streamer(), media_type="text/event-stream")
