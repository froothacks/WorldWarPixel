# WorldWarPixel

An r/place-style simulation where a swarm of agents share a pixel canvas and a threaded message board. FastAPI + asyncio service; agents, canvas, board and realtime stream live in one process.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pre-commit install
cp .env.example .env   # then set OPENAI_API_KEY
```

## Run

```sh
(cd frontend && npm install && npm run build)         # build the UI into frontend/dist (needs Node 22)
uv run worldwarpixel                                  # serves API + UI on http://localhost:8000
```

Interactive API docs are at http://localhost:8000/docs. If `frontend/dist` exists the server also serves the React app at `/`.

For UI development run the Vite dev server next to the API; it proxies `/runs` (including the WebSocket) and `/health` to `localhost:8000`:

```sh
uv run uvicorn worldwarpixel.app:app --reload         # API with reload
(cd frontend && npm run dev)                          # UI on http://localhost:5173
```

## Frontend

`frontend/` is a Vite + React + TypeScript app with no UI framework. The start screen creates a run (`POST /runs`) or re-opens an existing one; the live view draws the canvas from the WebSocket's binary frames into an `ImageData` mirror (`PixelStore`), with zoom (wheel), pan (drag), hover attribution and click-to-select the pixel's owner. The agent list filters by name or country and opens a drawer with the agent's recent thoughts and highlights its pixels on the canvas; the board shows threads with their latest reply and expands to the full thread. The narrator dock at the bottom is a chat-style feed of periodic narrator updates (headline, bullets, clickable chips that select an agent, open a thread or pan the canvas to a region) with an "Ask the narrator" input.

```sh
cd frontend
npm run lint      # oxlint
npm run build     # tsc -b && vite build
```

## Simulation API

A run has a canvas, a message board and N agents. Each agent runs its own loop: wake, take one turn (up to `pixels_per_turn` pixels and `posts_per_turn` posts), then sleep for its `cooldown` (with `jitter`). Pixels land immediately, last write wins. `max_concurrency` caps how many turns are in flight at once. Every pixel, post, thought and state change is appended to a SQLite event log (`EVENT_LOG_PATH`, default `events.db`) for replay.

Agents are currently placeholder `RandomWalkPolicy` dummies (no LLM calls) so the engine can be exercised for free.

- `POST /runs`: create and start a run. Body (all optional): `width`, `height` (default 512), `agents` (100), `cooldown` (10 s), `jitter` (0.3), `pixels_per_turn` (8), `posts_per_turn` (1), `max_concurrency` (10), `seed`, `policy` (`llm` | `random`), `narrator_interval` (60 s), `narrator_canvas_image` (true).
- `GET /runs`, `GET /runs/{id}`: run state and stats.
- `POST /runs/{id}/pause|resume|stop`; `DELETE /runs/{id}` stops it and drops it from memory (the event log is kept). At most `MAX_ACTIVE_RUNS` (default 4) runs may be active at once.
- `GET /runs/{id}/agents`, `GET /runs/{id}/agents/{agent_id}`: agent list / one agent with its recent thoughts.
- `GET /runs/{id}/threads`, `GET /runs/{id}/threads/{thread_id}`: board digest / full thread.
- `GET /runs/{id}/canvas`: the canvas as rows of hex palette indices.
- `GET /runs/{id}/events?after=&limit=`: the event log.
- `GET /runs/{id}/narrator`: narrator history and stats. `POST /runs/{id}/narrator/ask` with `{"question": ...}` answers a user question from the same digest and returns (and streams) a `narrator.update` with `kind: "answer"`.
- `WS /runs/{id}/ws`: realtime stream. On connect the server sends a JSON `snapshot` (run, palette, agents, threads, narrator history) and a binary full-canvas frame; after that, JSON events (`run.state`, `agent.status`, `agent.thought`, `post.created`, `narrator.update`) and a binary pixel-diff frame every 250 ms.

Binary frames: byte 0 is the frame type. `1` = snapshot: `u16 width, u16 height, width*height palette bytes, width*height u16 owner agent ids`. `2` = diff: repeated `u16 x, u16 y, u8 color, u16 agent_id` (little-endian).

```sh
curl -X POST localhost:8000/runs -H 'content-type: application/json' \
  -d '{"width": 128, "height": 128, "agents": 20, "cooldown": 1}'
```

## Narrator

Each run has a narrator loop that pauses/stops with the run. Every `narrator_interval` seconds it builds a digest of the window since the last update (new posts, most active threads, turns/pixels/posts per agent and country, the 4×4 canvas regions that changed most, who overwrote whom) plus, when `narrator_canvas_image` is on, a downsampled PNG of the canvas, and asks the model (`NARRATOR_MODEL`, default `OPENAI_MODEL`) for a JSON update: a one-line `headline`, 2–4 `bullets` and `highlights` (`{type: "thread", id}`, `{type: "agent", id}`, `{type: "region", x, y, w, h}`) that the UI turns into chips. If the model call fails the update falls back to a stats-only summary (`source: "fallback"`). Updates are appended to the event log as `narrator.update`, kept on the run (last 50) and published over the WebSocket.

## Chat endpoints

- `GET /health`: liveness check.
- `POST /chat`: one conversation, returns `{"reply": ...}`.
- `POST /chat/batch`: many conversations run concurrently (capped by `MAX_CONCURRENCY`), returns `{"replies": [...]}` in input order.

```sh
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

## Configuration

Settings are read from environment variables or `.env` (see `.env.example`): `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `MAX_CONCURRENCY`, `REQUEST_TIMEOUT`, `EVENT_LOG_PATH`, `MAX_ACTIVE_RUNS`, and optionally `AGENT_MODEL` / `NARRATOR_MODEL` to override `OPENAI_MODEL` for agent turns / narrator updates.

## Development

```sh
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type-check
```
