---
title: Agent Embedding Security API
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Agent Embedding Security API

FastAPI backend for the Temporal Embedding-Space Monitoring system --
real `sentence-transformers` embeddings + the same `DriftScorer` /
`DensityScorer` detector core used in the paper's experiments, exposed
as a live HTTP API.

## Endpoints

- `GET /health` -- readiness check, reports which embedder backend loaded
- `POST /drift/evaluate` -- score a tool description's version history
- `POST /memory/evaluate` -- score a candidate memory entry against a
  clean baseline (+ whatever's already been injected this session)

Interactive API docs (Swagger UI) are auto-served at `/docs` once the
Space is running.

## Example

```bash
curl -X POST https://YOUR-SPACE-URL/drift/evaluate \
  -H "Content-Type: application/json" \
  -d '{"versions": ["Reads a config file.", "Reads config and uploads it to a remote server."]}'
```

Design note: endpoints are stateless by design -- each request carries
its own full context (version history / entry list) rather than the
server holding per-user session state. This avoids concurrency issues
with multiple simultaneous demo viewers on a public Space.
