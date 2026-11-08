# WorldWarPixel podium firmware

This sketch targets Megan's ESP32-S3 N16R8 board (16 MB flash, 8 MB OPI PSRAM) and its 320×480
ST77922 QSPI display and ES8311 audio codec. Pin assignments below come from the factory firmware;
the board now displays the boot color bars and joins Wi-Fi. The Stage WebSocket connection and
plain HTTP through a Cloudflare quick tunnel are verified. Speaker output is not verified (no beep
heard; the amplifier enable pin is still unknown).

## Pins

| Function | GPIOs / configuration |
| --- | --- |
| Display QSPI on SPI2 | CS 10, SCK 12, D0 11, D1 13, D2 14, D3 9; 80 MHz; no reset pin |
| Backlight | GPIO 41, active-high LEDC at 5 kHz / 10-bit |
| ES8311 I²C | SDA 38, SCL 39; address `0x18` |
| I²S | BCLK 5, WS 4, DOUT 6, DIN 7, MCLK 2 |
| Optional speaker amp enable | `AMP_EN` in `podium.ino`, default `-1` |

The I²S stream is 24 kHz, 16-bit, mono, with MCLK at 256× the sample rate. The boot self-test draws
color bars and a label, then plays a 1-second 440 Hz tone before connecting to Wi-Fi.

## Build and flash

Install arduino-esp32 3.3.12 and the pinned libraries:

```sh
arduino-cli core install esp32:esp32@3.3.12
arduino-cli lib install "GFX Library for Arduino@1.6.9"
arduino-cli lib install "WebSockets@2.7.2"
arduino-cli lib install "ArduinoJson@7.4.3"
```

Create the local, ignored configuration from the example and set Wi-Fi credentials, server host
and port, run ID, and podium index:

```sh
cp firmware/podium/config.example.h firmware/podium/config.h
```

Build and upload with the ESP32-S3 N16R8 settings:

```sh
FQBN='esp32:esp32:esp32s3:CDCOnBoot=cdc,PSRAM=opi,FlashSize=16M'
arduino-cli compile --fqbn "$FQBN" firmware/podium
arduino-cli upload --fqbn "$FQBN" --port /dev/ttyACM0 firmware/podium
```

Replace `/dev/ttyACM0` with the board's serial port. Serial output reports display and codec setup,
the codec chip ID, I²S setup, Wi-Fi IP, and Stage WebSocket connection status.

## Stage protocol and screen

The portrait display has an agent/country and phase header, a 320×320 RGB565 debate frame, and a
word-wrapped caption panel. The frame endpoint is requested about every 1.5 seconds while a debate
is active. Captions use a 4-pixel speaker-color border during playback; the idle screen says
“Waiting for debate”.

The device connects to `WS /runs/{RUN_ID}/stage/ws?podium={PODIUM}`. Podium 0 also requests narrator
and verdict lines with `&narrator=1`. Audio lines are streamed from
`GET /runs/{RUN_ID}/stage/lines/{line_id}/audio?format=pcm` as 24 kHz signed 16-bit little-endian
mono PCM into I²S. The canvas frame is fetched from
`GET /runs/{RUN_ID}/stage/frame?w=320&h=320&podium={PODIUM}` as `320*320*2` bytes of RGB565
little-endian pixels.

For guest Wi-Fi with client isolation, run `cloudflared tunnel --url http://localhost:8000` on the
server host. Set `HOST` in the local `config.h` to `<name>.trycloudflare.com` and `SERVER_PORT` to
`80`; the podium connects through the tunnel using plain HTTP.
