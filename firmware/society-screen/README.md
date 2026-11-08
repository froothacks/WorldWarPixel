# Society Screen — WorldWarPixel

Firmware for the physical screens that stand next to the WorldWarPixel demo
(ESP-IDF; the older Arduino speech panel lives in `../podium`). Two modes,
switched with the SOCIETY / INTERVIEW tabs at the top right (the choice is kept
in NVS across reboots):

- **Society**: shows one society's live state (members, top board threads, the
  current murmur, canvas thumbnail). Hold the button, speak, and the server folds
  what you said into an unattributed murmur that agents of that society may notice.
- **Interview**: talk to one agent. Pick someone by tapping their name, or hold
  the button and say who ("let me talk to Imran", "someone from Peru"). Then hold
  the button to ask questions; the agent answers in character with the run's
  model, the reply is captioned on screen and spoken through the speaker in the
  agent's voice. Holding the button while it speaks interrupts it. The exchange is
  added to the agent's memory, so it may come up in what the agent does next.

The same binary works on all three screens. Only `DEVICE_TOKEN` differs; the
server maps token -> society index via `DEVICE_TOKENS="<tok0>:0,<tok1>:1,<tok2>:2"`.

## Board

LCDWIKI **ES3C35P** (also sold as "Hosyond 3.5\" ESP32-S3 touchscreen"):

- ESP32-S3, 16 MB flash, 8 MB octal PSRAM
- ST77922 320x480 display, QSPI (SCLK=12, D0-D3=11/13/14/9, CS=10, BL=41)
- ST7123 touch, I2C `0x55` (SDA=38, SCL=39, RST=48) — polled, no INT pin needed
- ES8311 codec w/ mic + speaker, I2C `0x18` (shared bus), I2S MCLK=17 BCLK=18 WS=21 DIN=16
  DOUT=15, full duplex at 16 kHz; speaker amp enable on GPIO1, active low
- WS2812 on GPIO40 (used as a status LED)

Note: the UI uses the project's own small framebuffer renderer rather than
LVGL — it is the same driver stack proven on this exact panel. The screen is
320x480 portrait (not 320x240 landscape).

## Toolchain

ESP-IDF **v5.5**, target `esp32s3`. Managed deps (component manager fetches
them): `esp_lcd_st77922`, `esp_lcd_touch_st7123`, `esp_websocket_client`,
`esp_codec_dev` (ES8311), `led_strip`.

## Configure

```
cp main/secrets.h.example main/secrets.h
# edit secrets.h: WIFI_SSID, WIFI_PASSWORD, SERVER_URL, DEVICE_TOKEN
```

`secrets.h` is gitignored. `SERVER_URL` has no trailing slash
(`http://192.168.1.50:8000` or `https://host` -> wss is derived automatically),
e.g. the Modal deployment `https://leonfattakhov--worldwarpixel-home.us-east.modal.run`.
The WebSocket sends the token both as `?token=` and as an `Authorization: Bearer`
header; HTTP requests use the header. Custom headers passed to
`esp_websocket_client` are inserted raw, so each line must end in `\r\n` or the
upgrade request never terminates ("Error read response for Upgrade header").

Reply audio needs the server's `TTS_API_KEY` (or `TTS_BASE_URL`); without it
interview replies are captions only.

## Build & flash

```
. ~/esp/esp-idf/export.sh
idf.py build
idf.py -p /dev/cu.usbmodemXXX flash monitor
```

Board gotcha: the LCD reset line is tied to CHIP_PU, so coming off other
firmware the panel can stay black until a real USB power-cycle. If the screen
is black after a clean boot log, unplug/replug the board.

## Mock server

Stdlib-only Python 3 (no pip installs):

```
python3 mock_server.py [8000]
DEVICE_TOKENS="alpha:0,bravo:1,charlie:2" python3 mock_server.py
```

Default tokens `alpha`/`bravo`/`charlie` -> societies 0/1/2. Set `SERVER_URL`
to your Mac's LAN IP (`ipconfig getifaddr en0`) and `DEVICE_TOKEN` to `alpha`
to see society 0 ("Ember"). It serves fake state every 2 s, migration flashes
~every 12 s, a random canvas thumbnail, and a canned transcript ~1.5 s after
`voice.end`, updating the murmur.

## Protocol coverage

- WS `/devices/ws`: hello (palette, society), state, migration toasts,
  ping/pong; push-to-talk streams `voice.start` + binary PCM frames
  (16 kHz mono int16 LE, 40 ms chunks) + `voice.end`, 20 s cap.
- Interview mode on the same socket: `agents.list` -> `agents`;
  `interview.select {"agent": id}` or a clip with `voice.start {"purpose": "pick"}`
  -> `interview.agent`; a clip with `{"purpose": "interview", "agent": id}` ->
  `interview.heard`, `interview.reply {text, audio, rate: 16000, bytes}`, binary
  PCM frames (16 kHz mono int16 LE), `interview.audio_end`; failures come back as
  `interview.error`.
- HTTP: `GET /devices/me` (token check -> "NOT PAIRED" screen with device MAC
  on 401), `GET /devices/canvas` (thumbnail polled ~5 s).
- Reconnect: Wi-Fi retry and WS retry both back off 1/2/5/10 s; UI stays
  alive offline and shows the status line.

## Layout

```
main/main.c     init, task wiring
main/hw.c       ST77922 QSPI display, ST7123 touch, WS2812, shared I2C bus
main/gfx.c      RGB565 framebuffer renderer (PSRAM, 5x7 font)
main/net.c      Wi-Fi station, WebSocket client, HTTP helpers, state parsing
main/mic.c      ES8311 + full-duplex I2S: PTT streaming task, speaker playback task
main/ui.c       320x480 layout: mode tabs; society view (header, societies, threads,
                canvas, murmur) and interview view (people picker, conversation); PTT
main/state.h    shared app state (mutex-guarded)
mock_server.py  stdlib-only mock of the society-mode protocol (no interview mode)
```
