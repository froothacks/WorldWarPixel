// WorldWarPixel podium: one ESP32-S3 per debater.
//
// Connects to WS /runs/{RUN_ID}/stage/ws, shows the debater's avatar colour and captions on an
// ST7789 / ILI9341 display (TFT_eSPI), and plays its spoken lines through an I2S amp such as a
// MAX98357A by streaming GET /runs/{RUN_ID}/stage/lines/{id}/audio?format=pcm
// (24 kHz signed 16-bit little-endian mono).
//
// Libraries: ArduinoJson 7, links2004/WebSockets, TFT_eSPI (configure User_Setup.h for your panel).
// Board: any ESP32-S3 with PSRAM; arduino-esp32 core 3.x (ESP_I2S.h).
// Untested reference sketch: check pins against your wiring.

#include <Arduino.h>
#include <ArduinoJson.h>
#include <ESP_I2S.h>
#include <HTTPClient.h>
#include <TFT_eSPI.h>
#include <WebSocketsClient.h>
#include <WiFi.h>

// ---- configure ------------------------------------------------------------------------
const char *WIFI_SSID = "your-ssid";
const char *WIFI_PASS = "your-password";
const char *HOST = "192.168.1.10";  // machine running `uv run worldwarpixel`
const uint16_t PORT = 8000;
const char *RUN_ID = "abcd1234";    // from the URL of the live view (?run=...)
const int PODIUM = 0;               // 0 = left debater, 1 = right debater
const bool PLAY_NARRATOR = PODIUM == 0;  // one device also voices the MC and the verdict

const int I2S_BCLK = 15, I2S_LRCK = 16, I2S_DOUT = 17;
// ----------------------------------------------------------------------------------------

TFT_eSPI tft;
WebSocketsClient ws;
I2SClass i2s;

struct Line {
  int id = -1;
  bool audio = false;
  int sampleRate = 24000;
  float duration = 0;
};

volatile bool pending = false;
Line next;
String debaterName = "", debaterCountry = "", phase = "idle";
uint16_t debaterColor = TFT_DARKGREY;

uint16_t rgb565(const char *hex) {
  if (!hex || hex[0] != '#') return TFT_DARKGREY;
  long v = strtol(hex + 1, nullptr, 16);
  return tft.color565((v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff);
}

void drawPodium(const String &caption, const String &emotion, bool speaking) {
  tft.fillScreen(TFT_BLACK);
  int w = tft.width();
  // Pixel avatar: a block of the debater's canvas colour, framed while speaking.
  tft.fillRect(w / 2 - 40, 12, 80, 80, debaterColor);
  if (speaking) tft.drawRect(w / 2 - 44, 8, 88, 88, TFT_WHITE);
  tft.setTextDatum(TC_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.drawString(debaterName.length() ? debaterName : "waiting for debate", w / 2, 102, 4);
  tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  tft.drawString(debaterCountry + (emotion.length() ? "  (" + emotion + ")" : ""), w / 2, 130, 2);
  tft.setTextDatum(TL_DATUM);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setCursor(8, 156, 2);
  tft.setTextWrap(true);
  tft.print(caption);
}

void onState(JsonDocument &doc) {
  phase = doc["phase"] | "idle";
  debaterName = "";
  debaterCountry = "";
  debaterColor = TFT_DARKGREY;
  for (JsonObject d : doc["debaters"].as<JsonArray>()) {
    if (d["podium"] == PODIUM) {
      debaterName = d["name"].as<String>();
      debaterCountry = d["country"].as<String>();
      debaterColor = rgb565(d["color_hex"]);
    }
  }
  drawPodium(phase == "voting" ? "Vote now!" : String(doc["topic"] | ""), "", false);
}

void onLine(JsonDocument &doc) {
  bool mine = !doc["podium"].isNull() && doc["podium"] == PODIUM;
  bool narrator = doc["podium"].isNull();
  if (!mine && !(narrator && PLAY_NARRATOR)) return;
  if (mine) drawPodium(doc["text"].as<String>(), doc["emotion"].as<String>(), true);
  next.id = doc["id"];
  next.audio = doc["audio"];
  next.sampleRate = doc["sample_rate"] | 24000;
  next.duration = doc["duration"] | 0.0;
  pending = true;  // played from loop(), not inside the websocket callback
}

void onWs(WStype_t type, uint8_t *payload, size_t len) {
  if (type == WStype_CONNECTED) Serial.println("stage ws connected");
  if (type != WStype_TEXT) return;
  JsonDocument doc;
  if (deserializeJson(doc, payload, len)) return;
  String t = doc["type"].as<String>();
  if (t == "stage.snapshot" || t == "stage.state") onState(doc);
  else if (t == "stage.line") onLine(doc);
  else if (t == "stage.verdict" && doc["winner_id"].is<int>()) drawPodium("The crowd has spoken.", "", false);
}

void playLine(const Line &line) {
  if (!line.audio) {  // server has no TTS: just hold the caption for the line's duration
    delay((uint32_t)(line.duration * 1000));
    return;
  }
  HTTPClient http;
  String url = "http://" + String(HOST) + ":" + PORT + "/runs/" + RUN_ID + "/stage/lines/" + line.id +
               "/audio?format=pcm";
  http.begin(url);
  if (http.GET() != 200) {
    http.end();
    return;
  }
  i2s.end();
  i2s.begin(I2S_MODE_STD, line.sampleRate, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
  WiFiClient *stream = http.getStreamPtr();
  int remaining = http.getSize();
  static uint8_t buf[2048];
  while (http.connected() && (remaining > 0 || remaining == -1)) {
    size_t n = stream->readBytes(buf, sizeof(buf));
    if (n == 0) break;
    i2s.write(buf, n);
    if (remaining > 0) remaining -= n;
    ws.loop();  // keep the stage socket alive while talking
  }
  http.end();
}

void setup() {
  Serial.begin(115200);
  tft.init();
  tft.setRotation(1);
  drawPodium("connecting...", "", false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) delay(250);
  i2s.setPins(I2S_BCLK, I2S_LRCK, I2S_DOUT);
  i2s.begin(I2S_MODE_STD, 24000, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
  ws.begin(HOST, PORT, String("/runs/") + RUN_ID + "/stage/ws");
  ws.onEvent(onWs);
  ws.setReconnectInterval(2000);
}

void loop() {
  ws.loop();
  if (pending) {
    pending = false;
    playLine(next);
  }
}
