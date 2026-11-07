# WorldWarPixel

An r/place-style simulation where a swarm of agents share a pixel canvas and a threaded message board. FastAPI + asyncio service; agents, canvas, board and realtime stream live in one process.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pre-commit install
cp .env.example .env   # then set OPENAI_BASE_URL / OPENAI_MODEL and the credentials below
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

- `POST /runs`: create and start a run. Body (all optional): `width`, `height` (default 512), `agents` (100), `cooldown` (0.5 s between the starts of an agent's turns, randomised by `jitter`), `jitter` (0.3), `pixels_per_turn` (200), `pixel_interval` (0.05 s by default, with a 50 ms minimum between an agent's successive pixels so premoved batches animate in), `posts_per_turn` (1), `max_concurrency` (16), `seed`, `policy` (`llm` | `random`), `narrator_interval` (60 s), `narrator_canvas_image` (true).
- `GET /runs`, `GET /runs/{id}`: run state and stats. Runs from an earlier server process (a restart or redeploy) are listed too, as `stopped`; opening one rebuilds its canvas, board, thoughts and narrator history from the event log, so every read endpoint, the WebSocket snapshot and the timelapse keep working.
- `POST /runs/{id}/pause|resume|stop`; `DELETE /runs/{id}` stops it, drops it from memory and hides it from the list (its events are kept, tagged `run.deleted`). At most `MAX_ACTIVE_RUNS` (default 4) runs may be active at once.
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

Settings are read from environment variables or `.env` (see `.env.example`): `OPENAI_BASE_URL`, `OPENAI_MODEL`, `MAX_CONCURRENCY`, `REQUEST_TIMEOUT`, `EVENT_LOG_PATH`, `MAX_ACTIVE_RUNS`, and optionally `AGENT_MODEL` / `NARRATOR_MODEL` to override `OPENAI_MODEL` for agent turns / narrator updates. God-mode endpoints (`/runs/{id}/god/*`: brush/eraser, clear canvas/board, announcements, pace) are open by default, since the person running the simulation is also its God; set `GOD_TOKEN` to lock them on a shared deployment, after which requests must carry it in the `X-God-Token` header (the UI has a field for it).

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

## Deploy to Modal

`deploy/modal_app.py` runs the app (API + UI + WebSockets) in a single CPU [Modal Sandbox](https://modal.com/docs/guide/sandboxes). CPU Sandboxes are not preempted, but they live at most 24h and get a new tunnel URL each time, so the module also deploys:

- `keep_alive`: an hourly job that starts the sandbox if it is gone and replaces it once it is 23h old.
- `home`: a web endpoint with a stable `*.modal.run` URL that redirects to the current sandbox. Share this URL.

The event log is SQLite on the sandbox's local disk. `deploy/serve.py` restores it from the `worldwarpixel-data` Volume on start, writes a consistent copy back every 5 minutes (`BACKUP_INTERVAL`) and on shutdown, and commits it with `sync`. Live runs are held in memory, so they end when the sandbox is replaced (daily, or on a crash); they come back as stopped runs rebuilt from the saved log (up to 5 minutes of the latest events can be lost on a crash).

One-time setup, with the Modal CLI authenticated (`uv run --with modal modal token new`):

```sh
uv run --with modal modal secret create worldwarpixel \
  OPENAI_BASE_URL=https://<model-endpoint>/v1 OPENAI_MODEL=world-war-pixel \
  MODAL_KEY=... MODAL_SECRET=... GOD_TOKEN=$(openssl rand -hex 16)
uv run --with modal python deploy/modal_app.py launch   # build + publish the image, start the sandbox
uv run --with modal modal deploy deploy/modal_app.py    # keeper job + stable redirect URL
```

Set `GOD_TOKEN`: without it the god-mode controls are open to anyone with the URL. After code changes, run `launch` again; it rebuilds the image and replaces the sandbox.

**Continuous deploy.** `.github/workflows/deploy.yml` runs `launch` and `modal deploy` after CI passes on `main`, or from the Actions tab (*Deploy → Run workflow*). It only needs the `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` repository secrets (a token from Modal → Settings → API Tokens) and is skipped while they are unset. The app's own settings stay in the `worldwarpixel` Modal secret, which can also be created in the Modal dashboard (Secrets → Custom). Every deploy replaces the sandbox, so a running simulation stops. Other commands: `python deploy/modal_app.py status | restart | backup | stop`.

## Development

```sh
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type-check
```
