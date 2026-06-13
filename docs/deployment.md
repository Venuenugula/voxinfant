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

## Deploy to Render (recommended — cloud build, free tier)
1. Push this repo to GitHub.
2. On https://render.com → **New → Web Service** → connect the repo.
3. Render detects `render.yaml` (Docker runtime). Accept defaults → **Create**.
4. Render builds the image in the cloud and gives you a public URL.
   Health check: `/health`. Auto-redeploys on every push.

## Deploy to Railway (alternative)
1. Push to GitHub. On https://railway.app → **New Project → Deploy from repo**.
2. Railway auto-detects the `Dockerfile`. It injects `$PORT` automatically.
3. Deploy → public URL under **Settings → Networking → Generate Domain**.

## Notes
- The trained artifacts in `models/` (~2.3 MB) are committed, so cloud builds
  have the model with no extra steps.
- To retrain before deploying: `python scripts/prepare_data.py` (needs
  `data/raw/<class>/*.wav`) then `python -m voxinfant.train`, then redeploy.
- CORS is currently open (`*`) for convenience — tighten `allow_origins` in
  `api/app.py` for production.
