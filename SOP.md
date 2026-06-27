# Second Brain SOP — Replicate for Any LinkedIn Creator

## Overview
Build a RAG chat app trained on someone's LinkedIn posts. The app answers questions in their voice using Claude Haiku, with TF-IDF retrieval over their own content.

**Architecture**: LinkedIn API (MCP) → scrape posts → store as `.md` → TF-IDF vectors → cosine similarity → Claude Haiku → chat UI

## Prerequisites

| Item | Why | Cost |
|------|-----|------|
| GitHub account | Host code, trigger deploys | Free |
| Render account | Host the app | Free tier (512MB RAM) |
| Anthropic API key | Power the LLM answers | Pay-as-you-go (~$0.30/chat) |
| LinkedIn account | Scrape posts from profiles | Free |
| This machine (or Codespace) | Run scraping scripts | Any Linux/macOS |

## Step 1 — LinkedIn Authentication

The LinkedIn MCP server uses browser automation (Chromium) to log in.

**First-time setup**: When you run any LinkedIn MCP tool, a Chromium browser will open. Log into LinkedIn there. The session is saved in `~/.linkedin-mcp/cookies.json`.

**To switch accounts**:
```bash
rm -rf ~/.linkedin-mcp/cookies.json ~/.linkedin-mcp/profile/Default
```
Then run a LinkedIn tool again — the browser will re-open for a fresh login.

## Step 2 — Scrape LinkedIn Posts

Use the `get_person_profile` tool from the LinkedIn MCP to scrape posts. The tool returns profile sections including posts.

Key parameters:
- `linkedin_username` — the public identifier from the profile URL (e.g., `rohitvirkud`)
- `sections` — set to `"posts"` to scrape only posts
- `max_scrolls` — how many scrolls deep to go (50+ posts needs ~5-10 scrolls)

Save each post as a numbered markdown file:
```
01_Title_of_first_post.md
02_Title_of_second_post.md
...
```

## Step 3 — Create the Project

### 3a. File structure
```
project_name/
├── app.py              # FastAPI web app
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deploy config
├── 01_*.md             # Scraped posts
├── 02_*.md
└── ...
```

### 3b. Dependencies (requirements.txt)
```
fastapi>=0.138.0
uvicorn[standard]>=0.34.0
scikit-learn>=1.3.0
anthropic>=0.84.0
numpy>=1.24.0
pydantic>=2.0.0
```

### 3c. App structure (app.py)
The app has four parts:

1. **Load posts** — reads all `[0-9]*.md` files in the directory
2. **Build TF-IDF index** — uses `sklearn.feature_extraction.text.TfidfVectorizer` — no model download, no OOM
3. **Retrieval** — `find_relevant_posts(query)` → transforms query → cosine similarity → returns top 4 posts
4. **Generation** — builds a prompt with persona + context → calls Claude Haiku → strips asterisks → returns answer

The persona prompt must be customized per person (see Step 4).

## Step 4 — Customize the Persona

Edit the `PERSONA_PROMPT` in `app.py`. Two sections to change:

**YOUR VOICE RULES** — generic style instructions (direct, conversational, tactical, etc.)

**YOUR CORE BELIEFS** — specific to the person. Read their posts and extract 4-6 recurring themes. Examples:
- Distribution > Content creation
- Flat fee pricing > AUM
- Inbound works: connect with ICPs → post → DM

## Step 5 — Customize the UI

Edit the inline HTML in `app.py` @app.get("/"):
- Title: change "Rohit's Second Brain" to "[Name]'s Second Brain"
- Greeting message: change the welcome text
- Colors: update the CSS variables (background, accent colors) to match the person's brand

## Step 6 — Deploy to Render

### 6a. Push to GitHub
```bash
cd project_name
git init && git add -A && git commit -m "Initial commit"
gh repo create <org>/<repo-name> --push --public
```

### 6b. Deploy via Render Dashboard
1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repo
4. Settings:
   - **Name**: `<person>-second-brain`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
5. Environment Variables:
   - `ANTHROPIC_API_KEY`: `sk-ant-...` (your key)

### 6c. Auto-deploy
Render auto-deploys on every push to the default branch.

## Step 7 — Verify

1. Wait for deploy to finish (status → "Live")
2. Open the URL (e.g., `https://<project>.onrender.com`)
3. Test the chat with questions relevant to the person's content
4. Check that:
   - Answers are in their voice
   - No asterisks appear
   - No source citations are shown
   - Responses are fast (<10s)

## Troubleshooting

### OOM on Render free tier (512MB)
**Symptoms**: Deploy fails with "Out of memory (used over 512Mi)"
**Cause**: PyTorch, sentence-transformers, or ONNX runtime models exceed 512MB
**Fix**: Use TF-IDF (scikit-learn) instead of embedding models. No model downloads needed, ~50MB total.

### "shapes not aligned" error
**Cause**: No posts loaded (empty embeddings matrix)
**Fix**: Check `POSTS_DIR` — on Render, files need to be in the same directory as `app.py`. Use `os.path.dirname(os.path.abspath(__file__))` as default.

### LinkedIn auth issues
**Symptoms**: MCP server returns errors about not being logged in
**Fix**: Delete `~/.linkedin-mcp/cookies.json` and run the tool again to trigger fresh login.

### API key not found
**Symptoms**: Chat returns "ANTHROPIC_API_KEY not set"
**Fix**: Set `ANTHROPIC_API_KEY` in Render dashboard Environment Variables, then deploy again.

## Appendix — Key Files

### app.py
The main application. Sections to modify per person:
- Line ~15: `POSTS_DIR` (if posts are elsewhere)
- Line ~48-68: `PERSONA_PROMPT` (voice, beliefs)
- Line ~106-253: HTML template (title, greeting, colors)

### requirements.txt
Dependencies. Do NOT add:
- `sentence-transformers` (pulls PyTorch, OOM on free tier)
- `fastembed` (pulls ONNX runtime, OOM on free tier)
- `torch` / `torchvision` (too heavy)

Use `scikit-learn` for TF-IDF instead.

### render.yaml
Optional Render blueprint. The dashboard UI is simpler.
