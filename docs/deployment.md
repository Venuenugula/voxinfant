# Deployment

The app is a **single service**: FastAPI serves both the JSON API and the static
UI, with the trained model baked in. The `Dockerfile` honours `$PORT`.

## Run locally (no Docker)
```bash
pip install -e .
PYTHONPATH=src uvicorn api.app:app --port 8000
# open http://localhost:8000
```

## Run locally with Docker
```bash
docker compose up --build
# open http://localhost:8000
```
> Note: building the image needs ~1–2 GB of free disk. If your disk is full,
> use the cloud-build path below instead (the platform builds the image).

## Deploy: backend on Hugging Face Spaces + frontend on Vercel

Vercel cannot run this Python/ML backend (250 MB serverless bundle limit, no
system libs like libsndfile, no Docker). So: **backend on HF Spaces (Docker),
frontend (static `web/`) on Vercel**, pointed at the Space.

### 1. Backend → Hugging Face Spaces (Docker)
1. https://huggingface.co/new-space → **SDK: Docker**, blank template, public.
2. Push this repo to the Space's git remote (it builds the `Dockerfile`):
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space>
   git push space main
   ```
   The Space's README YAML frontmatter sets `sdk: docker` and `app_port: 8000`.
3. Wait for the build; the Space serves at `https://<user>-<space>.hf.space`.
   That URL already runs the **full app** (UI + API) — usable on its own.

### 2. Frontend → Vercel (optional, static)
1. https://vercel.com → **Add New → Project** → import the GitHub repo.
2. **Root Directory = `web`**, Framework Preset = **Other**, no build command.
3. Deploy. Then point the UI at the backend once, by visiting:
   `https://<your-vercel-app>/?api=https://<user>-<space>.hf.space`
   (the backend URL is remembered in localStorage). Or set `DEFAULT_API_BASE`
   in `web/index.html` before deploying.

CORS is already open (`allow_origins=["*"]`), so the Vercel page can call the
Space. Tighten it to your Vercel domain for production.

## Other container hosts (Railway / Fly.io)
- **Railway:** New Project → Deploy from repo → auto-detects `Dockerfile`,
  injects `$PORT`. Generate a domain under Settings → Networking.
- **Fly.io:** `fly launch` (uses the `Dockerfile`) → `fly deploy`.
- **Render:** `render.yaml` is included (Docker runtime, `/health` check).

## Notes
- The trained artifacts in `models/` (~2.3 MB) are committed, so cloud builds
  have the model with no extra steps.
- To retrain before deploying: `python scripts/prepare_data.py` (needs
  `data/raw/<class>/*.wav`) then `python -m voxinfant.train`, then redeploy.
- CORS is currently open (`*`) for convenience — tighten `allow_origins` in
  `api/app.py` for production.
