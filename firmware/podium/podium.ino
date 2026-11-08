#include <Arduino.h>
#include <ArduinoJson.h>
#include <Arduino_GFX_Library.h>
#include <ESP_I2S.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <Wire.h>
#include <WebSocketsClient.h>
#include <esp_heap_caps.h>
#include <esp32-hal-psram.h>
#include <math.h>

#include "config.h"
#include "es8311.h"
#include "st77922_init.h"

constexpr int BACKLIGHT_PIN = 41;
const int AMP_EN = 1;  // active low
constexpr int I2S_BCLK = 18;
constexpr int I2S_WS = 21;
constexpr int I2S_DOUT = 15;
constexpr int I2S_DIN = 16;
constexpr int I2S_MCLK = 17;

constexpr int SCREEN_WIDTH = 320;
constexpr int SCREEN_HEIGHT = 480;
constexpr int HEADER_HEIGHT = 50;
constexpr int FRAME_HEIGHT = 320;
constexpr int CAPTION_Y = HEADER_HEIGHT + FRAME_HEIGHT;
constexpr int CAPTION_HEIGHT = SCREEN_HEIGHT - CAPTION_Y;
constexpr size_t FRAME_BYTES = SCREEN_WIDTH * FRAME_HEIGHT * 2;
constexpr uint16_t COLOR_BLACK = 0x0000;
constexpr uint16_t COLOR_WHITE = 0xFFFF;
constexpr uint16_t COLOR_GRAY = 0x7BEF;
constexpr uint16_t COLOR_DARK_GRAY = 0x4208;
constexpr uint32_t FRAME_REFRESH_MS = 1500;
constexpr uint8_t LINE_QUEUE_CAPACITY = 8;
constexpr size_t INIT_OPS_CAPACITY = 1024;

Arduino_ESP32QSPI displayBus(10, 12, 11, 13, 14, 9);
Arduino_ST77916 *display = nullptr;
WebSocketsClient stageSocket;
I2SClass i2s;
uint16_t *frameBuffer = nullptr;
uint8_t initOperations[INIT_OPS_CAPACITY];
size_t initOperationsLength = 0;

struct StageLine {
  int id = -1;
  bool audio = false;
  uint32_t sampleRate = 24000;
  float duration = 0;
  uint16_t color = COLOR_WHITE;
  String text;
};

StageLine lineQueue[LINE_QUEUE_CAPACITY];
uint8_t lineQueueHead = 0;
uint8_t lineQueueTail = 0;
uint8_t lineQueueSize = 0;
bool linePlaying = false;
bool speaking = false;
uint32_t lastFrameMs = 0;
String phase = "idle";
String debaterName;
String debaterCountry;
String currentCaption;
uint16_t debaterColor = COLOR_DARK_GRAY;
uint16_t captionColor = COLOR_WHITE;

bool appendOperation(uint8_t value, size_t &length) {
  if (length >= INIT_OPS_CAPACITY) return false;
  initOperations[length++] = value;
  return true;
}

bool buildInitOperations() {
  size_t length = 0;
  if (!appendOperation(BEGIN_WRITE, length)) return false;

  for (size_t i = 0; i < sizeof(ST77922_INIT) / sizeof(ST77922_INIT[0]); ++i) {
    const InitCmd &entry = ST77922_INIT[i];
    // QSPI needs command + params in one 0x02 transaction; WRITE_BYTES would send them as pixels.
    const bool ok = entry.len == 0
                        ? appendOperation(WRITE_COMMAND_8, length) && appendOperation(entry.cmd, length)
                        : appendOperation(WRITE_C8_BYTES, length) && appendOperation(entry.cmd, length) &&
                              appendOperation(entry.len, length);
    if (!ok) return false;
    for (uint8_t j = 0; j < entry.len; ++j) {
      if (!appendOperation(entry.data[j], length)) return false;
    }
    if (entry.delay_ms > 0) {
      if (entry.delay_ms > 255 ||
          !appendOperation(DELAY, length) ||
          !appendOperation(static_cast<uint8_t>(entry.delay_ms), length)) {
        return false;
      }
    }
  }

  if (!appendOperation(END_WRITE, length)) return false;
  initOperationsLength = length;
  return true;
}

uint16_t colorFromHex(const String &hex, uint16_t fallback) {
  if (display == nullptr || hex.length() < 7 || hex[0] != '#') return fallback;
  const uint32_t value = strtoul(hex.c_str() + 1, nullptr, 16);
  return display->color565((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF);
}

String wrapCaption(const String &text) {
  constexpr size_t MAX_CHARS_PER_LINE = 48;
  constexpr uint8_t MAX_LINES = 11;
  String output;
  String line;
  size_t start = 0;
  uint8_t lineCount = 0;

  while (start < text.length()) {
    int end = text.indexOf(' ', start);
    if (end < 0) end = text.length();
    String word = text.substring(start, end);
    if (line.length() > 0 && line.length() + 1 + word.length() > MAX_CHARS_PER_LINE) {
      if (output.length() > 0) output += '\n';
      output += line;
      line = word;
      if (++lineCount >= MAX_LINES) {
        output += "\n...";
        return output;
      }
    } else {
      if (line.length() > 0) line += ' ';
      line += word;
    }
    start = static_cast<size_t>(end) + 1;
  }

  if (line.length() > 0) {
    if (output.length() > 0) output += '\n';
    output += line;
  }
  return output;
}

void drawHeader() {
  if (display == nullptr) return;
  const uint16_t foreground =
      ((debaterColor & 0xF800) * 299UL / 0xF800 +
       ((debaterColor & 0x07E0) >> 5) * 587UL / 0x3F +
       (debaterColor & 0x001F) * 114UL / 0x1F) > 500
          ? COLOR_BLACK
          : COLOR_WHITE;
  display->fillRect(0, 0, SCREEN_WIDTH, HEADER_HEIGHT, debaterColor);
  display->setTextWrap(false);
  display->setTextSize(1);
  display->setTextColor(foreground);
  display->setCursor(8, 8);
  display->print(debaterName.length() > 0 ? debaterName : String("WorldWarPixel podium"));
  if (debaterCountry.length() > 0) {
    display->print(" - ");
    display->print(debaterCountry);
  }
  display->setCursor(8, 29);
  display->print("Phase: ");
  display->print(phase);
}

void drawCaption() {
  if (display == nullptr) return;
  display->fillRect(0, CAPTION_Y, SCREEN_WIDTH, CAPTION_HEIGHT, COLOR_BLACK);
  const uint16_t border = speaking ? captionColor : COLOR_DARK_GRAY;
  for (int inset = 0; inset < 4; ++inset) {
    display->drawRect(
        inset,
        CAPTION_Y + inset,
        SCREEN_WIDTH - inset * 2,
        CAPTION_HEIGHT - inset * 2,
        border);
  }
  display->setTextWrap(false);
  display->setTextSize(1);
  display->setTextColor(COLOR_WHITE);
  display->setCursor(12, CAPTION_Y + 10);
  const String text = currentCaption.length() > 0 ? currentCaption : "Waiting for debate";
  display->print(wrapCaption(text));
}

void showWaitingScreen() {
  if (display == nullptr) return;
  currentCaption = "Waiting for debate";
  display->fillRect(0, HEADER_HEIGHT, SCREEN_WIDTH, FRAME_HEIGHT, COLOR_BLACK);
  drawCaption();
}

bool appendLine(const StageLine &line) {
  if (lineQueueSize >= LINE_QUEUE_CAPACITY) {
    Serial.println("stage line queue full");
    return false;
  }
  lineQueue[lineQueueHead] = line;
  lineQueueHead = (lineQueueHead + 1) % LINE_QUEUE_CAPACITY;
  ++lineQueueSize;
  return true;
}

bool takeLine(StageLine &line) {
  if (lineQueueSize == 0) return false;
  line = lineQueue[lineQueueTail];
  lineQueueTail = (lineQueueTail + 1) % LINE_QUEUE_CAPACITY;
  --lineQueueSize;
  return true;
}

void applyStageState(JsonDocument &doc) {
  const String oldPhase = phase;
  phase = doc["phase"] | "idle";
  bool foundDebater = false;

  for (JsonObjectConst debater : doc["debaters"].as<JsonArrayConst>()) {
    if (!debater["podium"].isNull() && debater["podium"].as<int>() == PODIUM) {
      debaterName = debater["name"] | "";
      debaterCountry = debater["country"] | "";
      debaterColor = colorFromHex(debater["color_hex"] | "", COLOR_DARK_GRAY);
      foundDebater = true;
      break;
    }
  }

  if (!foundDebater && phase == "idle") {
    debaterName = "";
    debaterCountry = "";
    debaterColor = COLOR_DARK_GRAY;
  }
  drawHeader();

  if (oldPhase == "idle" && phase != "idle") lastFrameMs = millis() - FRAME_REFRESH_MS;
  if (phase == "idle" && !linePlaying && lineQueueSize == 0) showWaitingScreen();
}

void queueStageLine(JsonDocument &doc) {
  const String kind = doc["kind"] | "";
  const bool isMine = !doc["podium"].isNull() && doc["podium"].as<int>() == PODIUM;
  const bool isNarrator = doc["podium"].isNull() && (kind == "narrator" || kind == "verdict");
  if (!isMine && !(PLAY_NARRATOR && isNarrator)) return;

  StageLine line;
  line.id = doc["id"] | -1;
  line.audio = doc["audio"] | false;
  line.sampleRate = doc["sample_rate"] | 24000;
  line.duration = doc["duration"] | 0.0f;
  line.text = doc["text"] | "";
  line.color = colorFromHex(doc["color_hex"] | "", isMine ? debaterColor : COLOR_WHITE);
  appendLine(line);
}

void onStageEvent(WStype_t type, uint8_t *payload, size_t length) {
  if (type == WStype_CONNECTED) {
    Serial.println("ws connected");
    return;
  }
  if (type == WStype_DISCONNECTED) {
    Serial.println("ws disconnected");
    return;
  }
  if (type != WStype_TEXT) return;

  JsonDocument doc;
  if (deserializeJson(doc, payload, length)) return;
  const String eventType = doc["type"] | "";
  if (eventType == "stage.snapshot" || eventType == "stage.state") {
    applyStageState(doc);
  } else if (eventType == "stage.line") {
    queueStageLine(doc);
  }
}

void refreshFrame() {
  if (display == nullptr || frameBuffer == nullptr || WiFi.status() != WL_CONNECTED) return;
  String url = "http://" + String(HOST) + ":" + String(SERVER_PORT) +
               "/runs/" + String(RUN_ID) + "/stage/frame?w=320&h=320&podium=" + String(PODIUM);
  HTTPClient http;
  http.setTimeout(5000);
  http.begin(url);
  const int status = http.GET();
  if (status != HTTP_CODE_OK) {
    Serial.printf("stage frame http %d\n", status);
    http.end();
    return;
  }

  const int contentLength = http.getSize();
  if (contentLength >= 0 && static_cast<size_t>(contentLength) != FRAME_BYTES) {
    Serial.printf("stage frame size %d, expected %u\n", contentLength, FRAME_BYTES);
    http.end();
    return;
  }

  WiFiClient *stream = http.getStreamPtr();
  size_t received = 0;
  const uint32_t deadline = millis() + 5000;
  while (received < FRAME_BYTES && static_cast<int32_t>(millis() - deadline) < 0) {
    const int available = stream->available();
    if (available > 0) {
      size_t wanted = static_cast<size_t>(available);
      if (wanted > FRAME_BYTES - received) wanted = FRAME_BYTES - received;
      const size_t count = stream->readBytes(
          reinterpret_cast<char *>(reinterpret_cast<uint8_t *>(frameBuffer) + received),
          wanted);
      received += count;
    } else if (!http.connected()) {
      break;
    } else {
      delay(1);
    }
  }
  http.end();

  if (received == FRAME_BYTES) {
    display->draw16bitRGBBitmap(0, HEADER_HEIGHT, frameBuffer, SCREEN_WIDTH, FRAME_HEIGHT);
  } else {
    Serial.printf("stage frame truncated: %u/%u\n", received, FRAME_BYTES);
  }
}

void serviceFrameRefresh() {
  if (phase != "idle" && millis() - lastFrameMs >= FRAME_REFRESH_MS) {
    lastFrameMs = millis();
    refreshFrame();
  }
}

void waitForLine(float seconds) {
  const uint32_t durationMs = static_cast<uint32_t>(seconds * 1000.0f);
  const uint32_t started = millis();
  while (millis() - started < durationMs) {
    stageSocket.loop();
    serviceFrameRefresh();
    delay(2);
  }
}

void playAudio(const StageLine &line) {
  if (line.sampleRate != 24000) {
    Serial.printf("unsupported stage sample rate: %u\n", line.sampleRate);
    waitForLine(line.duration);
    return;
  }

  String url = "http://" + String(HOST) + ":" + String(SERVER_PORT) +
               "/runs/" + String(RUN_ID) + "/stage/lines/" + String(line.id) + "/audio?format=pcm";
  HTTPClient http;
  http.setTimeout(5000);
  http.begin(url);
  if (http.GET() != HTTP_CODE_OK) {
    Serial.printf("stage audio unavailable for line %d\n", line.id);
    http.end();
    waitForLine(line.duration);
    return;
  }

  int remaining = http.getSize();
  WiFiClient *stream = http.getStreamPtr();
  uint8_t buffer[2048];
  while ((remaining > 0 || remaining == -1) && (remaining > 0 || http.connected())) {
    const int available = stream->available();
    if (available <= 0) {
      if (!http.connected()) break;
      stageSocket.loop();
      delay(2);
      continue;
    }

    size_t wanted = sizeof(buffer);
    if (wanted > static_cast<size_t>(available)) wanted = static_cast<size_t>(available);
    if (remaining > 0 && wanted > static_cast<size_t>(remaining)) {
      wanted = static_cast<size_t>(remaining);
    }
    const size_t count = stream->readBytes(reinterpret_cast<char *>(buffer), wanted);
    if (count == 0) break;
    i2s.write(buffer, count);
    if (remaining > 0) remaining -= count;
    stageSocket.loop();
  }
  http.end();
}

void playNextLine() {
  StageLine line;
  if (!takeLine(line)) return;
  linePlaying = true;
  speaking = true;
  currentCaption = line.text;
  captionColor = line.color;
  drawCaption();

  if (line.audio) playAudio(line);
  else waitForLine(line.duration);

  speaking = false;
  linePlaying = false;
  drawCaption();
  serviceFrameRefresh();
  if (phase == "idle" && lineQueueSize == 0) showWaitingScreen();
}

void drawSelfTest() {
  if (display == nullptr) return;
  constexpr uint16_t bars[] = {
      0xF800, 0x07E0, 0x001F, 0xFFE0, 0x07FF, 0xF81F, 0xFFFF, 0x7BEF};
  const int barWidth = SCREEN_WIDTH / (sizeof(bars) / sizeof(bars[0]));
  display->fillScreen(COLOR_BLACK);
  for (size_t i = 0; i < sizeof(bars) / sizeof(bars[0]); ++i) {
    display->fillRect(i * barWidth, 0, barWidth, SCREEN_HEIGHT, bars[i]);
  }
  display->fillRect(0, SCREEN_HEIGHT / 2 - 24, SCREEN_WIDTH, 48, COLOR_BLACK);
  display->setTextSize(2);
  display->setTextColor(COLOR_WHITE);
  display->setCursor(24, SCREEN_HEIGHT / 2 - 8);
  display->print("WorldWarPixel podium");
}

void playSelfTestTone() {
  constexpr uint32_t SAMPLE_RATE = 24000;
  constexpr float TWO_PI_VALUE = 6.28318530718f;
  constexpr float AMPLITUDE = 6000.0f;
  // Ode to Joy, opening phrase.
  static const float notes[] = {329.63f, 329.63f, 349.23f, 392.0f, 392.0f, 349.23f, 329.63f,
                                293.66f, 261.63f, 261.63f, 293.66f, 329.63f, 329.63f, 293.66f,
                                293.66f};
  static const uint8_t beats[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 1, 4};
  constexpr uint32_t BEAT_SAMPLES = SAMPLE_RATE / 6;
  uint8_t buffer[512];

  for (size_t n = 0; n < sizeof(notes) / sizeof(notes[0]); ++n) {
    const uint32_t total = BEAT_SAMPLES * beats[n];
    const uint32_t fade = SAMPLE_RATE / 100;
    for (uint32_t offset = 0; offset < total;) {
      const uint32_t samplesPerChunk = sizeof(buffer) / 2;
      const uint32_t left = total - offset;
      const uint32_t count = samplesPerChunk < left ? samplesPerChunk : left;
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t k = offset + i;
        float env = 1.0f;
        if (k < fade) env = static_cast<float>(k) / fade;
        if (total - k < fade * 4) env = static_cast<float>(total - k) / (fade * 4);
        const float angle = TWO_PI_VALUE * notes[n] * k / SAMPLE_RATE;
        const int16_t sample = static_cast<int16_t>(AMPLITUDE * env * sinf(angle));
        buffer[i * 2] = sample & 0xFF;
        buffer[i * 2 + 1] = (sample >> 8) & 0xFF;
      }
      i2s.write(buffer, count * 2);
      offset += count;
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);

  if (!buildInitOperations()) {
    Serial.println("display init table overflow");
  } else {
    display = new Arduino_ST77916(
        &displayBus,
        -1,
        0,
        true,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        0,
        0,
        0,
        0,
        initOperations,
        initOperationsLength);
    if (display != nullptr && display->begin(80000000)) {
      Serial.println("display ok");
    } else {
      Serial.println("display init failed");
    }
  }

  pinMode(BACKLIGHT_PIN, OUTPUT);
  ledcAttach(BACKLIGHT_PIN, 5000, 10);
  ledcWrite(BACKLIGHT_PIN, 1023);

  Wire.begin(38, 39);
  Wire.setClock(400000);
  uint8_t chipId1 = 0;
  uint8_t chipId2 = 0;
  if (es8311ReadChipId(chipId1, chipId2)) {
    Serial.printf("codec id 0x%02X 0x%02X\n", chipId1, chipId2);
  } else {
    Serial.println("codec id read failed");
  }
  if (es8311Init()) Serial.println("codec init ok");
  else Serial.println("codec init failed");

  if (AMP_EN >= 0) {
    pinMode(AMP_EN, OUTPUT);
    digitalWrite(AMP_EN, LOW);
  }

  i2s.setPins(I2S_BCLK, I2S_WS, I2S_DOUT, I2S_DIN, I2S_MCLK);
  const bool i2sReady =
      i2s.begin(I2S_MODE_STD, 24000, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
  if (i2sReady) Serial.println("I2S ok");
  else Serial.printf("I2S init failed: %d\n", i2s.lastError());

  if (display != nullptr) drawSelfTest();
  if (i2sReady) playSelfTestTone();

  frameBuffer = static_cast<uint16_t *>(ps_malloc(FRAME_BYTES));
  if (frameBuffer == nullptr) Serial.println("PSRAM frame buffer allocation failed");

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) delay(250);
  Serial.print("wifi ip ");
  Serial.println(WiFi.localIP());

  const String path = String("/runs/") + RUN_ID + "/stage/ws?podium=" + PODIUM +
                      (PLAY_NARRATOR ? "&narrator=1" : "");
  stageSocket.begin(HOST, SERVER_PORT, path);
  stageSocket.onEvent(onStageEvent);
  stageSocket.setReconnectInterval(2000);
}

void loop() {
  stageSocket.loop();
  serviceFrameRefresh();
  if (!linePlaying && lineQueueSize > 0) playNextLine();
  delay(2);
}
