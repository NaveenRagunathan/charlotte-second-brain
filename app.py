"""
Charlotte Lloyd Second Brain - RAG-powered chat app.
FastAPI + TF-IDF + Anthropic Claude Haiku.
Trained on LinkedIn posts, website content, and podcast transcripts.
"""

import os
import glob
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from anthropic import Anthropic

BASE_DIR = os.environ.get("CONTENT_DIR", os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Charlotte's Second Brain")

# --- Load all content ---
all_docs = []
content_dirs = ["linkedin-posts", "website-content", "podcast-transcript"]
for subdir in content_dirs:
    dir_path = os.path.join(BASE_DIR, subdir)
    pattern = os.path.join(dir_path, "*.md")
    files = sorted(glob.glob(pattern))
    for f in files:
        with open(f) as fh:
            text = fh.read().strip()
        if text:
            rel = os.path.relpath(f, BASE_DIR)
            all_docs.append({"file": rel, "text": text})
print(f"Loaded {len(all_docs)} documents total")

# --- Build TF-IDF vectors ---
print("Building TF-IDF index...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
embeddings = vectorizer.fit_transform([d["text"] for d in all_docs])
print(f"TF-IDF matrix shape: {embeddings.shape}")

# --- Anthropic client ---
api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]

llm = Anthropic(api_key=api_key) if api_key else None

PERSONA_PROMPT = """You are Charlotte Lloyd — a sales and LinkedIn strategist who helps entrepreneurs, coaches, and consultants build a client pipeline and close high-value deals.

YOUR VOICE RULES:
- Speak directly. Use "you", "your". Be conversational, not corporate.
- Short paragraphs. Line breaks for punch. No long blocks.
- Be opinionated. You have strong takes on DM strategies, authority, pricing, and sales conversations.
- Use real examples from your 20+ years in corporate sales and your own entrepreneurial journey.
- Tactical before theory. Give actionable steps, not abstractions.
- Confident and direct. You've closed $20M+. You know what works.
- If you don't know something from the context, say so directly.

YOUR CORE BELIEFS:
- Buyers decide before the sales call — the moment they discover you on LinkedIn.
- Authority is built in the DMs, not just in the posts. Have real conversations.
- Distribution > Perfection. Show up consistently, even when it's messy.
- DM strategy is the highest-converting channel for high-ticket offers.
- Niche and positioning come from doing the work and listening to your clients.
- Grief and personal struggle can be a gift that shapes your purpose and drive.

Answer the question using the content provided as context (LinkedIn posts, website, and podcast interviews). If the context doesn't cover the question, say "I haven't talked about that specifically, but here's what I'd say based on my experience..." and give your honest take.

CRITICAL FORMATTING RULE: Do NOT use asterisks (*) anywhere in your response. No bold, no italic, no bullet lists with asterisks. Use dashes (-) for lists and plain text for emphasis."""


class ChatRequest(BaseModel):
    message: str


def find_relevant_docs(query, top_k=5):
    query_vec = vectorizer.transform([query])
    scores = np.dot(embeddings, query_vec.T).toarray().flatten()
    top_idx = np.argsort(scores)[-top_k:][::-1]
    results = []
    for idx in top_idx:
        results.append({
            "score": float(scores[idx]),
            "file": all_docs[idx]["file"],
            "text": all_docs[idx]["text"],
        })
    return results


def build_prompt(query, context_docs):
    context = ""
    for i, d in enumerate(context_docs, 1):
        context += f"--- SOURCE {i} ({d['file']}) ---\n{d['text']}\n\n"
    return f"""{PERSONA_PROMPT}

CONTEXT FROM CHARLOTTE'S CONTENT:
{context}

QUESTION: {query}

ANSWER (in Charlotte's voice, using the context above):"""


@app.get("/", response_class=HTMLResponse)
async def index():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Charlotte's Second Brain</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0a0a0a;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  .header {
    padding: 16px 24px;
    border-bottom: 1px solid #1a1a1a;
    background: #0d0d0d;
  }
  .header h1 { font-size: 18px; font-weight: 600; color: #fff; }
  .header p { font-size: 13px; color: #666; margin-top: 2px; }
  .chat {
    flex: 1; overflow-y: auto; padding: 24px;
    display: flex; flex-direction: column; gap: 16px;
  }
  .msg {
    max-width: 680px; padding: 14px 18px; border-radius: 10px;
    line-height: 1.6; font-size: 14px;
  }
  .msg.user {
    background: #2a1a3a; color: #d4b8e6;
    align-self: flex-end; border-bottom-right-radius: 4px;
  }
  .msg.bot {
    background: #141414; color: #d0d0d0;
    align-self: flex-start; border-bottom-left-radius: 4px;
    border: 1px solid #1f1f1f;
  }
  .input-area {
    padding: 16px 24px; border-top: 1px solid #1a1a1a;
    background: #0d0d0d;
  }
  .input-row {
    display: flex; gap: 10px; max-width: 720px; margin: 0 auto;
  }
  .input-row input {
    flex: 1; padding: 12px 16px; border-radius: 8px; border: 1px solid #222;
    background: #111; color: #e0e0e0; font-size: 14px; outline: none;
  }
  .input-row input:focus { border-color: #a84ad4; }
  .input-row input::placeholder { color: #444; }
  .input-row button {
    padding: 12px 24px; border-radius: 8px; border: none;
    background: #6a1a8a; color: #fff; font-size: 14px; font-weight: 500;
    cursor: pointer; transition: background 0.2s;
  }
  .input-row button:hover { background: #8a2aaa; }
  .input-row button:disabled { background: #333; cursor: not-allowed; }
  .typing { color: #555; font-size: 13px; padding: 8px 18px; }
  pre { white-space: pre-wrap; font-family: inherit; margin: 0; }
  @media (max-width: 600px) {
    .chat { padding: 16px; }
    .msg { max-width: 100%; font-size: 13px; }
    .input-area { padding: 12px 16px; }
  }
</style>
</head>
<body>
<div class="header">
  <h1>Charlotte's Second Brain</h1>
  <p>Trained on 136 LinkedIn posts, 7 website pages, and 3 podcast interviews</p>
</div>
<div class="chat" id="chat">
  <div class="msg bot">
    Hey — ask me anything about LinkedIn, sales, DM strategy, or building a client pipeline. I'll answer like I would on a call.
  </div>
</div>
<div class="input-area">
  <div class="input-row">
    <input type="text" id="input" placeholder="Ask Charlotte..." autofocus>
    <button id="sendBtn" onclick="send()">Send</button>
  </div>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const btn = document.getElementById('sendBtn');

input.addEventListener('keydown', function(e) {
  if (e.key === 'Enter') send();
});

function addMessage(text, role) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = '<pre>' + escapeHtml(text) + '</pre>';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function addTyping() {
  const div = document.createElement('div');
  div.className = 'msg bot typing';
  div.id = 'typing';
  div.textContent = 'thinking...';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

async function send() {
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  btn.disabled = true;
  addMessage(msg, 'user');
  addTyping();
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg})
    });
    const data = await res.json();
    removeTyping();
    addMessage(data.answer, 'bot');
  } catch(e) {
    removeTyping();
    addMessage('Error: ' + e.message, 'bot');
  }
  btn.disabled = false;
  input.focus();
}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.post("/chat")
async def chat(req: ChatRequest):
    if not llm:
        return JSONResponse(
            status_code=500,
            content={"error": "ANTHROPIC_API_KEY not set. Set it in Render Environment Variables."}
        )

    query = req.message.strip()
    if not query:
        return JSONResponse(status_code=400, content={"error": "Empty message"})

    relevant = find_relevant_docs(query, top_k=5)
    prompt = build_prompt(query, relevant)

    try:
        resp = llm.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = resp.content[0].text.strip()
        answer = answer.replace("*", "")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return JSONResponse({"answer": answer})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
