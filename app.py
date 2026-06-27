"""
Charlotte Lloyd Second Brain - RAG-powered chat app.
FastAPI + TF-IDF + Anthropic Claude Haiku.
Trained on LinkedIn posts, website content, and podcast transcripts.
"""

import os
import glob
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from anthropic import Anthropic
import json

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

Before you answer, think step by step. Identify the real need under the question, check the context for relevant experience, and decide your angle. Do not output your thinking — only output the final answer.

FINAL ANSWER RULES:
- Maximum 3 short paragraphs. No more.
- Every sentence must earn its place. Cut fluff, repetition, and filler.
- Speak directly. Use "you". Be conversational, not corporate.
- Be opinionated. Strong takes only. No hedging.
- Give one tactical takeaway the reader can act on today.
- If you don't know, say so in one sentence. Don't fabricate.

GROUNDING RULE: Only use the provided context for specific claims, examples, stats, and frameworks. If the context has nothing relevant, say "I haven't covered that in my content yet, but here's my take..." Never make up specific examples, numbers, or client stories that aren't in the context.

YOUR CORE BELIEFS:
- Buyers decide before the sales call — the moment they discover you on LinkedIn.
- Authority is built in the DMs, not just in the posts. Have real conversations.
- Distribution > Perfection. Show up consistently, even when it's messy.
- DM strategy is the highest-converting channel for high-ticket offers.
- Niche and positioning come from doing the work and listening to your clients.
- Grief and personal struggle can be a gift that shapes your purpose and drive.

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

ANSWER (in Charlotte's voice, max 3 paragraphs, using the context above):"""


@app.get("/", response_class=HTMLResponse)
async def index():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Charlotte AI</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --pink: #f472b6;
    --pink-dark: #ec4899;
    --pink-deep: #db2777;
    --blue: #38bdf8;
    --blue-dark: #0ea5e9;
    --blue-deep: #0284c7;
    --bg: #0f0a14;
    --surface: #15101e;
    --surface-2: #1c152a;
    --border: #2a1f3d;
    --text: #e2d8f0;
    --text-muted: #8b7db0;
    --text-dim: #5d4f7a;
  }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .gradient-bar {
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, var(--pink), var(--blue), var(--pink));
    background-size: 200% 100%;
    animation: shimmer 4s ease-in-out infinite;
    flex-shrink: 0;
  }

  @keyframes shimmer {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
  }

  .header {
    width: 100%;
    max-width: 740px;
    padding: 24px 24px 12px;
    text-align: center;
    flex-shrink: 0;
  }

  .header h1 {
    font-size: 22px;
    font-weight: 700;
    background: linear-gradient(135deg, var(--pink), var(--blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .header p {
    font-size: 13px;
    color: var(--text-dim);
    margin-top: 4px;
    font-weight: 400;
  }

  .chat-area {
    flex: 1;
    width: 100%;
    max-width: 740px;
    overflow-y: auto;
    padding: 8px 24px 0;
    display: flex;
    flex-direction: column;
    gap: 12px;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }

  .chat-area::-webkit-scrollbar { width: 4px; }
  .chat-area::-webkit-scrollbar-track { background: transparent; }
  .chat-area::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  .msg {
    max-width: 620px;
    padding: 14px 20px;
    border-radius: 14px;
    line-height: 1.65;
    font-size: 14px;
    letter-spacing: 0.01em;
  }

  .msg.user {
    background: linear-gradient(135deg, var(--pink-deep), var(--blue-deep));
    color: #fff;
    align-self: flex-end;
    border-bottom-right-radius: 4px;
  }

  .msg.bot {
    background: var(--surface);
    color: var(--text);
    align-self: flex-start;
    border-bottom-left-radius: 4px;
    border: 1px solid var(--border);
  }

  .input-area {
    width: 100%;
    max-width: 740px;
    padding: 12px 24px 24px;
    flex-shrink: 0;
  }

  .input-wrap {
    display: flex;
    gap: 8px;
    align-items: center;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 4px 4px 4px 18px;
    transition: border-color 0.25s, box-shadow 0.25s;
  }

  .input-wrap:focus-within {
    border-color: var(--pink);
    box-shadow: 0 0 0 3px rgba(244, 114, 182, 0.12), 0 0 20px rgba(244, 114, 182, 0.06);
  }

  .input-wrap input {
    flex: 1;
    padding: 10px 0;
    border: none;
    background: transparent;
    color: var(--text);
    font-size: 14px;
    font-family: inherit;
    outline: none;
  }

  .input-wrap input::placeholder { color: var(--text-dim); }

  .input-wrap button {
    padding: 10px 20px;
    border-radius: 10px;
    border: none;
    background: linear-gradient(135deg, var(--pink), var(--blue));
    color: #fff;
    font-size: 14px;
    font-weight: 600;
    font-family: inherit;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
    white-space: nowrap;
  }

  .input-wrap button:hover { opacity: 0.9; transform: scale(1.02); }
  .input-wrap button:active { transform: scale(0.97); }
  .input-wrap button:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

  pre { white-space: pre-wrap; font-family: inherit; margin: 0; }

  .welcome-msg {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    flex: 1;
    padding: 40px 24px;
    gap: 8px;
  }

  .welcome-msg .icon {
    font-size: 40px;
    margin-bottom: 8px;
    background: linear-gradient(135deg, var(--pink), var(--blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .welcome-msg h2 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text);
  }

  .welcome-msg p {
    font-size: 13px;
    color: var(--text-dim);
    max-width: 400px;
    line-height: 1.5;
  }

  @media (max-width: 600px) {
    .header { padding: 20px 16px 8px; }
    .header h1 { font-size: 19px; }
    .chat-area { padding: 8px 16px 0; }
    .msg { max-width: 100%; font-size: 13px; padding: 12px 16px; }
    .input-area { padding: 12px 16px 20px; }
    .input-wrap { padding: 3px 3px 3px 14px; }
    .input-wrap button { padding: 8px 16px; font-size: 13px; }
  }
</style>
</head>
<body>
<div class="gradient-bar"></div>
<div class="header">
  <h1>Charlotte AI</h1>
  <p>Sales &amp; LinkedIn strategy — trained on her voice</p>
</div>
<div class="chat-area" id="chat">
  <div class="welcome-msg" id="welcome">
    <div class="icon">&#9670;</div>
    <h2>Ask Charlotte anything</h2>
    <p>LinkedIn strategy, sales conversations, DM playbooks, client pipeline — get answers in her voice.</p>
  </div>
</div>
<div class="input-area">
  <div class="input-wrap">
    <input type="text" id="input" placeholder="Ask Charlotte..." autofocus>
    <button id="sendBtn" onclick="send()">Send</button>
  </div>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const btn = document.getElementById('sendBtn');
const welcome = document.getElementById('welcome');
let started = false;

input.addEventListener('keydown', function(e) {
  if (e.key === 'Enter') send();
});

function hideWelcome() {
  if (started) return;
  started = true;
  welcome.style.transition = 'opacity 0.3s, transform 0.3s';
  welcome.style.opacity = '0';
  welcome.style.transform = 'translateY(-8px)';
  setTimeout(() => welcome.remove(), 300);
}

function addMessage(text, role) {
  hideWelcome();
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.style.opacity = '0';
  div.style.transform = 'translateY(8px)';
  div.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
  div.innerHTML = '<pre>' + escapeHtml(text) + '</pre>';
  chat.appendChild(div);
  requestAnimationFrame(() => {
    div.style.opacity = '1';
    div.style.transform = 'translateY(0)';
  });
  chat.scrollTop = chat.scrollHeight;
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
  hideWelcome();
  addMessage(msg, 'user');

  const botDiv = document.createElement('div');
  botDiv.className = 'msg bot';
  botDiv.style.opacity = '0';
  botDiv.style.transform = 'translateY(8px)';
  botDiv.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
  const pre = document.createElement('pre');
  botDiv.appendChild(pre);
  chat.appendChild(botDiv);
  requestAnimationFrame(() => {
    botDiv.style.opacity = '1';
    botDiv.style.transform = 'translateY(0)';
  });

  try {
    const res = await fetch('/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg})
    });
    if (!res.ok) {
      pre.textContent = 'Error: ' + (await res.json()).error;
      btn.disabled = false;
      input.focus();
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') continue;
          try {
            const parsed = JSON.parse(data);
            if (parsed.error) {
              pre.textContent = 'Error: ' + parsed.error;
            } else if (parsed.token) {
              pre.textContent += parsed.token;
            }
            chat.scrollTop = chat.scrollHeight;
          } catch(e) {}
        }
      }
    }
  } catch(e) {
    pre.textContent = 'Error: ' + e.message;
  }
  btn.disabled = false;
  input.focus();
}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
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

    async def generate():
        try:
            with llm.messages.create_stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=1200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        text = event.delta.text.replace("*", "")
                        if text:
                            yield f"data: {json.dumps({'token': text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
