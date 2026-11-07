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

1. `WS /runs/{id}/stage/ws`: a `stage.snapshot`, then JSON `stage.state`, `stage.line`,
   `stage.votes`, `stage.reaction` and `stage.verdict` events.
2. For each `stage.line` whose `podium` matches (narrator/verdict lines have `podium: null`), if
   `audio` is true fetch `GET /runs/{id}/stage/lines/{line_id}/audio?format=pcm`: raw signed 16-bit
   little-endian mono at `sample_rate` (24 kHz), streamed straight to I2S. Otherwise show the caption
   for `duration` seconds.
