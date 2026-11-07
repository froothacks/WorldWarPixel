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

## Persona data

Agents are seeded to be representative of the world population and each one speaks as a specific
survey respondent (`src/worldwarpixel/personas/`):

1. **Seats per country** come from UN World Population Prospects 2024 (medium variant, 2024) via
   largest-remainder apportionment, so 100 agents give 18 India, 17 China, 4 USA, 4 Indonesia, …
   A compact derived table (`src/worldwarpixel/data/wpp2024_population.csv`, UN data is public) is
   committed; refresh it with `uv run worldwarpixel-personas fetch-wpp`.
2. **Each seat is one World Values Survey Wave 7 respondent** from that country, sampled with the
   survey weight (`W_WEIGHT`). Age, sex, education, employment/occupation, income decile, religion and
   religiosity, left–right self-placement (Q240), generalised trust (Q57), attitudes to democracy
   (Q250/Q235/Q238), country-coded ethnic group (Q290) etc. become the agent's **persona card**
   (`AgentSpec.persona_card`, shown in the UI drawer) and a ~5-sentence **persona text** written as
   the respondent's own answers ("you place yourself at 4 on a 1 (left) to 10 (right) scale") that
   opens the LLM prompt. Countries WVS did not survey borrow respondents from the nearest surveyed
   country in the same cultural zone (`personas/proxy.py`; the card shows `survey_country`).
3. **Names** come from per-country, sex-matched name pools (`sim/names.py`); optional
   [Nemotron-Personas](https://huggingface.co/nvidia) data adds a concrete occupation and hobbies.

### Downloading the WVS microdata (optional, recommended)

WVS data is free but requires registration and **may not be redistributed**, so it is not in git
(`/data/` is ignored) and the app falls back to a **clearly-labelled synthetic dataset**
(`src/worldwarpixel/data/synthetic_respondents.csv`, `source=synthetic`, hand-written priors per
country — it is *not* survey data; a warning is logged at startup when it is used).

1. Register at https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp and download the
   *WVS Cross-National Wave 7* **CSV** release (v6.0; a zip containing one `WVS_Cross-National_Wave_7_csv_v6_0.csv`), unzip it.
2. Import it (keeps only the variables above, one row per respondent, ~95k rows):

   ```sh
   uv run worldwarpixel-personas import-wvs ~/Downloads/WVS_Cross-National_Wave_7_csv_v6_0.csv
   # -> data/wvs7_respondents.csv  (override with $WVS_RESPONDENTS_PATH)
   ```

3. Optionally fetch Nemotron-Personas samples (CC BY 4.0) for occupation/hobbies enrichment:

   ```sh
   uv run worldwarpixel-personas fetch-nemotron          # USA IND SGP GBR -> data/nemotron/
   ```

4. Inspect the allocation and sample personas, or regenerate the synthetic fallback:

   ```sh
   uv run worldwarpixel-personas show --agents 100 --sample 5
   uv run worldwarpixel-personas show --json --sample 1   # full persona cards
   uv run worldwarpixel-personas build-synthetic
   ```

`seed_agents()` picks up `data/wvs7_respondents.csv` automatically on the next run (no restart needed); survey countries
missing from the file are topped up from the synthetic set. Generated persona pools derived from WVS
must stay out of git.

Sources: UN DESA, World Population Prospects 2024; Haerpfer et al. (eds.), World Values Survey Wave 7
(2017–2022) Cross-National Data-Set v6.0, doi:10.14281/18241.24; NVIDIA Nemotron-Personas (CC BY 4.0).

## Development

```sh
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type-check
```
