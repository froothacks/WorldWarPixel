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
uv run worldwarpixel                                  # serves on http://localhost:8000
uv run uvicorn worldwarpixel.app:app --reload         # dev server with reload
```

Interactive API docs are at http://localhost:8000/docs.

## Simulation API

A run has a canvas, a message board and N agents. Each agent runs its own loop: wake, take one turn (up to `pixels_per_turn` pixels and `posts_per_turn` posts), then sleep for its `cooldown` (with `jitter`). Pixels land immediately, last write wins. `max_concurrency` caps how many turns are in flight at once. Every pixel, post, thought and state change is appended to a SQLite event log (`EVENT_LOG_PATH`, default `events.db`) for replay.

Agents are currently placeholder `RandomWalkPolicy` dummies (no LLM calls) so the engine can be exercised for free.

- `POST /runs`: create and start a run. Body (all optional): `width`, `height` (default 512), `agents` (100), `cooldown` (10 s), `jitter` (0.3), `pixels_per_turn` (8), `posts_per_turn` (1), `max_concurrency` (10), `seed`.
- `GET /runs`, `GET /runs/{id}`: run state and stats.
- `POST /runs/{id}/pause|resume|stop`.
- `GET /runs/{id}/agents`, `GET /runs/{id}/agents/{agent_id}`: agent list / one agent with its recent thoughts.
- `GET /runs/{id}/threads`, `GET /runs/{id}/threads/{thread_id}`: board digest / full thread.
- `GET /runs/{id}/canvas`: the canvas as rows of hex palette indices.
- `GET /runs/{id}/events?after=&limit=`: the event log.
- `WS /runs/{id}/ws`: realtime stream. On connect the server sends a JSON `snapshot` (run, palette, agents, threads) and a binary full-canvas frame; after that, JSON events (`run.state`, `agent.status`, `agent.thought`, `post.created`) and a binary pixel-diff frame every 250 ms.

Binary frames: byte 0 is the frame type. `1` = snapshot: `u16 width, u16 height, width*height palette bytes`. `2` = diff: repeated `u16 x, u16 y, u8 color, u16 agent_id` (little-endian).

```sh
curl -X POST localhost:8000/runs -H 'content-type: application/json' \
  -d '{"width": 128, "height": 128, "agents": 20, "cooldown": 1}'
```

## Chat endpoints

- `GET /health`: liveness check.
- `POST /chat`: one conversation, returns `{"reply": ...}`.
- `POST /chat/batch`: many conversations run concurrently (capped by `MAX_CONCURRENCY`), returns `{"replies": [...]}` in input order.

```sh
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

## Configuration

Settings are read from environment variables or `.env` (see `.env.example`): `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `MAX_CONCURRENCY`, `REQUEST_TIMEOUT`, `EVENT_LOG_PATH`.

## Development

```sh
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type-check
```
