# WorldWarPixel

A simulation where a swarm of agents share a pixel canvas and a threaded message board. FastAPI + asyncio service; agents, canvas, board and realtime stream live in one process.

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

A run has a canvas, a message board and N agents. Each agent runs its own loop: wake, take one turn (up to `pixels_per_turn` pixels and `posts_per_turn` posts), then sleep for its `cooldown` (with `jitter`). Pixels land immediately, last write wins. `max_concurrency` caps how many turns are in flight at once. Every pixel, post, thought and state change is appended to the event log (`EVENT_LOG_PATH`, default `events.db`, or Postgres when `DATABASE_URL` is set) for replay. Per-run counters are maintained in a `runs` table for fast `/runs` listing; versioned schema migrations run automatically at startup and can also be applied with `python -m worldwarpixel.sim.migrations`.

Agents are currently placeholder `RandomWalkPolicy` dummies (no LLM calls) so the engine can be exercised for free.

- `POST /runs`: create and start a run. Body (all optional): `width`, `height` (default 512), `agents` (100), `cooldown` (0.5 s between the starts of an agent's turns, randomised by `jitter`), `jitter` (0.3), `pixels_per_turn` (200), `pixel_interval` (0.01 s by default, with a 10 ms minimum between an agent's successive pixels so premoved batches animate in), `posts_per_turn` (1), `max_concurrency` (1000; effectively one model call in flight per agent), `seed`, `policy` (`llm` | `random`), `narrator_interval` (60 s), `narrator_canvas_image` (true).
- `GET /runs`, `GET /runs/{id}`: run state and stats. Runs from an earlier server process (a restart or redeploy) are listed too, as `stopped`; opening one rebuilds its canvas, board, thoughts and narrator history from the event log, so every read endpoint, the WebSocket snapshot and the timelapse keep working.
- `POST /runs/{id}/pause|resume|stop`; `DELETE /runs/{id}` stops it, drops it from memory and hides it from the list (its events are kept, tagged `run.deleted`). At most `MAX_ACTIVE_RUNS` (default 4) runs may be active at once.
- `GET /runs/{id}/agents`, `GET /runs/{id}/agents/{agent_id}`: agent list / one agent with its recent thoughts.
- `GET /runs/{id}/threads`, `GET /runs/{id}/threads/{thread_id}`: board digest / full thread.
- `GET /runs/{id}/canvas`: the canvas as rows of hex palette indices.
- `GET /runs/{id}/events?after=&limit=`: the event log.
- `POST /runs/{id}/video`, `GET /runs/{id}/video`, `GET /runs/{id}/video/meta`, `GET /videos`: render / download / describe / list timelapse videos (see below).
- `GET /runs/{id}/narrator`: narrator history and stats. `POST /runs/{id}/narrator/ask` with `{"question": ...}` answers a user question from the same digest and returns (and streams) a `narrator.update` with `kind: "answer"`.
- `WS /runs/{id}/ws`: realtime stream. On connect the server sends a JSON `snapshot` (run, palette, agents, threads, narrator history) and a binary full-canvas frame; after that, JSON events (`run.state`, `agent.status`, `agent.thought`, `post.created`, `narrator.update`) and a binary pixel-diff frame every 250 ms.

Binary frames: byte 0 is the frame type. `1` = snapshot: `u16 width, u16 height, width*height palette bytes, width*height u16 owner agent ids`. `2` = diff: repeated `u16 x, u16 y, u8 color, u16 agent_id` (little-endian).

```sh
curl -X POST localhost:8000/runs -H 'content-type: application/json' \
  -d '{"width": 128, "height": 128, "agents": 20, "cooldown": 1}'
```

## Narrator

Each run has a narrator loop that pauses/stops with the run. Every `narrator_interval` seconds it builds a digest of the window since the last update (new posts, most active threads, turns/pixels/posts per agent and country, the 4×4 canvas regions that changed most, who overwrote whom) plus, when `narrator_canvas_image` is on, a downsampled PNG of the canvas, and asks the model (`NARRATOR_MODEL`, default `OPENAI_MODEL`) for a JSON update: a one-line `headline`, 2–4 `bullets` and `highlights` (`{type: "thread", id}`, `{type: "agent", id}`, `{type: "region", x, y, w, h}`) that the UI turns into chips. If the model call fails the update falls back to a stats-only summary (`source: "fallback"`). Updates are appended to the event log as `narrator.update`, kept on the run (last 50) and published over the WebSocket.

## Speech panel

Runs can speak short agent lines and read narrator updates aloud to the ESP32 panel in
[`firmware/podium/`](firmware/podium/). Configure `TTS_API_KEY` for OpenAI speech or
`TTS_BASE_URL` for an OpenAI-compatible service; `TTS_MODEL` defaults to `gpt-4o-mini-tts`.
The run option `speak_narrator` defaults to `true`.

- `POST /runs/{id}/speak` with `{"text": "...", "agent_id": 1}` queues a line (God-token protected).
- `GET /runs/{id}/speech` returns panel status, recent lines and synthesis stats.
- `GET /runs/{id}/speech/lines/{line_id}/audio?format=pcm` returns s16le mono PCM; WAV is the default.
- `WS /runs/{id}/speech/ws` streams a snapshot followed by `speech.line` events.

## Timelapse videos

Stopping a run (`POST /runs/{id}/stop`) renders an mp4 in the background (`TIMELAPSE_AUTO`, default on): the canvas timelapse on the left, the message board on the right with posts appearing on the same clock, and the latest narrator headline as a caption (`sim/video.py`; frames are composed with Pillow and piped to `ffmpeg`, which must be on the PATH or at `FFMPEG_PATH`). `POST /runs/{id}/video` renders on demand (also for live runs and runs restored from the log), `GET /runs/{id}/video` streams the file, `GET /videos` lists everything rendered. While a run is live, the video is re-rendered every `TIMELAPSE_AUTOSAVE_INTERVAL` seconds (default 30, 0 disables), each autosave overwriting the same `{run_id}.mp4`/`.json` so the archive always has one current video per run. Videos land in `TIMELAPSE_DIR` (`/data/timelapses` on the Modal Volume in production) next to a `{run_id}.json` sidecar with the run's config, stats, top countries and narrator headlines, so the archive can be scanned later to pick out the good ones. Set `TIMELAPSE_S3_BUCKET` (plus `TIMELAPSE_S3_ENDPOINT` — `https://storage.googleapis.com` for Google Cloud Storage, or the R2 endpoint — and `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, which for GCS are a service account's HMAC key) to mirror both files into a bucket. `python -m worldwarpixel.videos [run_id ...]` renders missing videos for archived runs.

## Chat endpoints

- `GET /health`: liveness check.
- `POST /chat`: one conversation, returns `{"reply": ...}`.
- `POST /chat/batch`: many conversations run concurrently (capped by `MAX_CONCURRENCY`), returns `{"replies": [...]}` in input order.

```sh
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

## Configuration

Settings are read from environment variables or `.env` (see `.env.example`): `OPENAI_BASE_URL`, `OPENAI_API_KEY` (optional; only sent to bearer-authenticated endpoints like real OpenAI, never to Modal), `OPENAI_MODEL`, `MAX_CONCURRENCY`, `REQUEST_TIMEOUT`, `EVENT_LOG_PATH` (or `DATABASE_URL` for a Postgres event log), `MAX_ACTIVE_RUNS`, the per-endpoint model secrets listed below (`MODAL_KEY` / `MODAL_SECRET`, `CHRYZTOPH_MODAL_*`, `MEGANWCHUN_MODAL_*`), and `AGENT_MAX_TOKENS` / `NARRATOR_MAX_TOKENS` (fallback output budgets, default 700, for model presets that don't carry their own; raise them for reasoning models that spend tokens thinking before they call a tool or finish the narrator JSON). God-mode endpoints (`/runs/{id}/god/*`: brush/eraser, clear canvas/board, announcements, pace) are open by default, since the person running the simulation is also its God; set `GOD_TOKEN` to lock them on a shared deployment, after which requests must carry it in the `X-God-Token` header (the UI has a field for it).

Set `LOG_LEVEL` to control application logging (default `INFO`); HTTP requests are logged with method, path, status and duration, and requests taking at least one second are warnings. Event-log reads taking at least one second also emit warnings with the run and pagination parameters.

Each run picks its model from the presets in `src/worldwarpixel/models.py` (the start screen shows them, `GET /models` lists them with whether this server has their credentials and per-endpoint call/error/failover counts): `gpt-6-luna` (OpenAI API, needs `OPENAI_API_KEY`), `qwen` (served from two Modal workspaces: advait-m with `MODAL_KEY` / `MODAL_SECRET` and chryztoph with `CHRYZTOPH_MODAL_KEY` / `CHRYZTOPH_MODAL_SECRET`), `deepseek` (Modal endpoint in the advait-m workspace, needs `MODAL_KEY` / `MODAL_SECRET`), and `glm` (GLM 5.3 Flash served from three Modal workspaces: advait-m with `MODAL_KEY` / `MODAL_SECRET`, chryztoph with `CHRYZTOPH_MODAL_KEY` / `CHRYZTOPH_MODAL_SECRET`, meganwchun with `MEGANWCHUN_MODAL_KEY` / `MEGANWCHUN_MODAL_SECRET`). Endpoints and the secret names each one needs are defined in code next to the preset; a preset is available when at least one of its endpoints has its secrets, and a run's calls are spread over every such endpoint (`src/worldwarpixel/routing.py`). Each agent (and each run's narrator) is pinned to one endpoint by rendezvous hashing so its conversation prefix stays in that server's KV cache; on a 429, 5xx (e.g. a cold Modal 503) or connection error the call fails over to the agent's next endpoint and the failing one cools down for 15 s. A preset also carries its API quirks — e.g. gpt-6-luna requires `max_completion_tokens`, no `chat_template_kwargs` and `reasoning_effort="none"` with tools — so adding a model or another deployment of one means adding a preset or an endpoint there.

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

`deploy/modal_app.py` serves the FastAPI app (API + UI + WebSockets) in-process in one always-on, non-preemptible [Modal web container](https://modal.com/docs/guide/webhooks), without a uvicorn subprocess or proxy hop for WebSockets. It serves the app at a stable `*.modal.run` URL; there is no 24-hour Sandbox lifetime cap. A single WebSocket connection is limited to 24 hours. The app uses exactly one container because live run state is held in memory.

Non-preemptible compute costs 3x the list price for CPU and memory, and the one container stays on continuously. The event log is SQLite on the container's local disk. `deploy/serve.py` restores it from the `worldwarpixel-data` Volume on start, writes a consistent copy back every 60 seconds (`BACKUP_INTERVAL`) and on shutdown, and commits it with `sync`. A deploy replaces the container, so live runs stop and reopen as stopped runs from the saved log. Deploys may briefly overlap old and new containers, so up to 60 seconds of events from the old container can be lost.

One-time setup, with the Modal CLI authenticated (`uv run --with modal==1.6.3 modal token new`):

```sh
uv run --with modal==1.6.3 modal secret create worldwarpixel \
  OPENAI_BASE_URL=https://<model-endpoint>/v1 OPENAI_MODEL=world-war-pixel \
  MODAL_KEY=... MODAL_SECRET=... GOD_TOKEN=$(openssl rand -hex 16)
uv run --with modal==1.6.3 modal deploy deploy/modal_app.py
```

Set `GOD_TOKEN`: without it the god-mode controls are open to anyone with the URL. To retire the legacy `worldwarpixel-web` Sandbox once, run `uv run --with modal==1.6.3 python deploy/modal_app.py retire`; it backs up the event log before stopping the Sandbox.

**Continuous deploy.** `.github/workflows/deploy.yml` runs the legacy-Sandbox `retire` step and then `modal deploy deploy/modal_app.py` after CI passes on `main`, or from the Actions tab (*Deploy → Run workflow*). It only needs the `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` repository secrets (a token from Modal → Settings → API Tokens) and is skipped while they are unset. The app's own settings stay in the `worldwarpixel` Modal secret, which can also be created in the Modal dashboard (Secrets → Custom). Every deploy replaces the container, so a running simulation stops.

### Cloud Run (GCP)

The same app also deploys to one always-on [Cloud Run](https://cloud.google.com/run) instance (`min=max=1`, CPU always allocated) in project `world-war-pixel-514101`. `Dockerfile` mirrors the Modal image (frontend built in, Whisper weights baked, ffmpeg) and `deploy/cloudrun.sh` builds it with Cloud Build and deploys it with the `world-war-pixel-timelapses-514101` bucket mounted at `/data`, so rendered timelapses land directly in GCS with no extra credentials. The SQLite backup only happens when Cloud SQL is skipped (`CLOUDSQL_INSTANCE=""`); with `DATABASE_URL` set, `deploy/serve.py` skips SQLite restore and backup. The app runs as the `worldwarpixel-run` service account, which gets `Storage Object Admin` on that bucket and `Secret Manager Secret Accessor` on the project. Cloud Run reads only the secrets allowlisted in `APP_SECRETS` in `deploy/cloudrun.sh`, exposing each by name as the same environment variable at version `latest`. Add a secret to both `APP_SECRETS` and Secret Manager to make Cloud Run mount it; secrets outside the allowlist are ignored. A project owner must grant the runtime service account `roles/secretmanager.secretAccessor` and the deployer service account `github-deployer@` `roles/secretmanager.viewer` on the project; the deploy script does not grant IAM roles. One-time setup with the gcloud CLI authenticated: `APP_ENV_FILE=/path/outside/repo/app.env deploy/cloudrun.sh`, where the optional file holds only non-secret `KEY=value` settings such as `OPENAI_BASE_URL` or `MAX_ACTIVE_RUNS`; it prints the service URL. `deploy.yml` runs the same script after CI on `main` when the `GCP_SA_KEY` repository secret (a deployer service-account JSON key with Cloud Run Admin, Artifact Registry Admin, Storage Admin, Service Account Admin, Service Account User, Cloud Build Editor, Service Usage Admin, and `roles/secretmanager.viewer`) is set. The optional `GCP_APP_ENV` repository secret is only for non-secret settings; app secrets are read from Secret Manager by name. The Modal deploy continues to use the `worldwarpixel` Modal secret.

## Development

```sh
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type-check
```
