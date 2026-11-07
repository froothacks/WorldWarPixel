# Podium firmware (ESP32-S3)

A reference client for the live debate stage: one device per debater. It is a second client of the
same protocol the browser podium (`/?stage=<run>&podium=0`) uses, so you can prototype in the browser
and swap in hardware later. Untested on real hardware.

Hardware: ESP32-S3 with PSRAM, an ST7789/ILI9341 SPI display (TFT_eSPI) and a MAX98357A I2S amp
plus speaker (default pins BCLK 15, LRCK 16, DOUT 17). An ESP32-S3-BOX-3 works with its own pins.

Libraries: ArduinoJson 7, `links2004/WebSockets`, TFT_eSPI; arduino-esp32 core 3.x.

Set `WIFI_SSID`, `WIFI_PASS`, `HOST`, `RUN_ID` and `PODIUM` (0 or 1) at the top of `podium.ino`.
Podium 0 also voices the narrator and the verdict.

Protocol:

1. Connect to `WS /runs/{id}/stage/ws?podium={PODIUM}&narrator={0|1}`. The server forwards only
   that podium's debate lines; `narrator=1` additionally forwards narrator and verdict lines
   (podium 0 should set it). Snapshots, `stage.state`, `stage.votes` and `stage.verdict` are always
   forwarded. The connected-device counts are included in stage state.
2. For each `stage.line`, if `audio` is true fetch
   `GET /runs/{id}/stage/lines/{line_id}/audio?format=pcm`: raw signed 16-bit little-endian mono at
   `sample_rate` (24 kHz), streamed straight to I2S. Otherwise show the caption for `duration`
   seconds.
3. Request `GET /runs/{id}/stage/frame?w=320&h=240&podium={PODIUM}` for a display frame. The
   response is exactly `w*h*2` bytes of RGB565 little-endian with `X-Width` and `X-Height` headers.
   The debate region (or the full canvas when idle) is nearest-neighbour scaled and letterboxed;
   debater-owned pixels are full color and other pixels are dimmed. The `podium` parameter is
   reserved and currently does not change the frame.
