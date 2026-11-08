#include "mic.h"
#include "hw.h"
#include "net.h"
#include "state.h"

#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "driver/i2s_std.h"
#include "driver/i2c_master.h"

#include "esp_codec_dev.h"
#include "esp_codec_dev_defaults.h"
#include "es8311_codec.h"

#include <string.h>
#include "driver/gpio.h"
#include "esp_heap_caps.h"

static const char *TAG = "mic";

/* ES8311 / board audio pins (vendor pinout) */
#define PIN_I2S_MCLK  17
#define PIN_I2S_BCLK  18
#define PIN_I2S_WS    21
#define PIN_I2S_DOUT  15   /* ESP -> codec (speaker) */
#define PIN_AMP_EN    1    /* speaker amp, active low */
#define PIN_I2S_DIN   16   /* codec -> ESP (mic) */
/* NB: esp_codec_dev wants the address in 8-bit form (it shifts right):
 * 7-bit 0x18 -> 0x30, which is also ES8311_CODEC_DEFAULT_ADDR. */
#define ES8311_ADDR   ES8311_CODEC_DEFAULT_ADDR

#define SAMPLE_RATE   16000
#define CHUNK_MS      40
#define CHUNK_SAMPLES (SAMPLE_RATE * CHUNK_MS / 1000)   /* 640 */
#define MAX_CLIP_MS   20000
#define SPK_MAX       (60 * SAMPLE_RATE * 2)   /* longest reply clip: 60 s (PSRAM) */
#define SPK_PREBUF    (SAMPLE_RATE * 2 / 4)    /* start playing after 250 ms */

static esp_codec_dev_handle_t s_dev;
static i2s_chan_handle_t s_rx;
static i2s_chan_handle_t s_tx;
static uint8_t *s_spk;                 /* reply clip, PSRAM */
static volatile size_t s_spk_wr;
static volatile bool s_spk_done;
static volatile uint32_t s_spk_gen;    /* bumps on every new clip / stop */
static volatile bool s_ptt;
static volatile bool s_have_mic;

/* simple voice processing: DC removal + fixed gain + clip */
static int32_t s_dc;

static void process(int16_t *s, int n, int *level)
{
    int64_t peak = 0;
    for (int i = 0; i < n; i++) {
        s_dc += ((int32_t)s[i] - s_dc) >> 8;      /* slow DC tracker */
        int32_t v = ((int32_t)s[i] - s_dc) * 4;   /* ~+12 dB */
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        s[i] = (int16_t)v;
        int32_t a = v < 0 ? -v : v;
        if (a > peak) peak = a;
    }
    *level = (int)((peak * 100) / 32768);
}

bool mic_init(void)
{
    /* I2S RX channel */
    gpio_config_t amp = {.pin_bit_mask = 1ULL << PIN_AMP_EN, .mode = GPIO_MODE_OUTPUT};
    gpio_config(&amp);
    gpio_set_level(PIN_AMP_EN, 1);   /* amp off until a reply plays */
    s_spk = heap_caps_malloc(SPK_MAX, MALLOC_CAP_SPIRAM);

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.auto_clear_after_cb = true;   /* send silence when nothing is queued */
    if (i2s_new_channel(&chan_cfg, &s_tx, &s_rx) != ESP_OK) {
        ESP_LOGE(TAG, "i2s channel alloc failed");
        return false;
    }
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT,
                                                        I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = PIN_I2S_MCLK,
            .bclk = PIN_I2S_BCLK,
            .ws = PIN_I2S_WS,
            .dout = PIN_I2S_DOUT,
            .din = PIN_I2S_DIN,
            .invert_flags = {0},
        },
    };
    if (i2s_channel_init_std_mode(s_rx, &std_cfg) != ESP_OK ||
        i2s_channel_init_std_mode(s_tx, &std_cfg) != ESP_OK) {
        ESP_LOGE(TAG, "i2s std init failed");
        return false;
    }

    /* the ES8311's I2C interface needs MCLK running before it answers;
     * start the channel now (codec data-if re-enable is tolerated). */
    i2s_channel_enable(s_rx);
    i2s_channel_enable(s_tx);

    /* codec over the shared i2c bus (0x18 sits next to touch 0x55) */
    audio_codec_i2c_cfg_t i2c_cfg = {
        .addr = ES8311_ADDR,
        .bus_handle = hw_i2c_bus(),
    };
    const audio_codec_ctrl_if_t *ctrl_if = audio_codec_new_i2c_ctrl(&i2c_cfg);
    if (!ctrl_if) {
        ESP_LOGE(TAG, "i2c ctrl if failed");
        return false;
    }

    audio_codec_i2s_cfg_t i2s_cfg = {
        .rx_handle = s_rx,
        .tx_handle = s_tx,
    };
    const audio_codec_data_if_t *data_if = audio_codec_new_i2s_data(&i2s_cfg);
    if (!data_if) {
        ESP_LOGE(TAG, "i2s data if failed");
        return false;
    }

    const audio_codec_gpio_if_t *gpio_if = audio_codec_new_gpio();
    es8311_codec_cfg_t es_cfg = {
        .codec_mode = ESP_CODEC_DEV_WORK_MODE_BOTH,
        .ctrl_if = ctrl_if,
        .gpio_if = gpio_if,
        .pa_pin = -1,              /* amp (GPIO1) is switched by spk_task */
        .use_mclk = true,
    };
    const audio_codec_if_t *codec_if = es8311_codec_new(&es_cfg);
    if (!codec_if) {
        ESP_LOGE(TAG, "es8311 init failed - voice disabled");
        return false;
    }

    esp_codec_dev_cfg_t dev_cfg = {
        .codec_if = codec_if,
        .data_if = data_if,
        .dev_type = ESP_CODEC_DEV_TYPE_IN_OUT,
    };
    s_dev = esp_codec_dev_new(&dev_cfg);
    if (!s_dev) {
        ESP_LOGE(TAG, "codec dev alloc failed");
        return false;
    }

    esp_codec_dev_sample_info_t fs = {
        .sample_rate = SAMPLE_RATE,
        .channel = 1,
        .bits_per_sample = 16,
    };
    if (esp_codec_dev_open(s_dev, &fs) != 0) {
        ESP_LOGE(TAG, "codec open failed");
        return false;
    }
    esp_codec_dev_set_in_gain(s_dev, 30.0);
    esp_codec_dev_set_out_vol(s_dev, 80);

    ESP_LOGI(TAG, "ES8311 mic + speaker ready @16kHz");
    s_have_mic = true;
    return true;
}

void mic_ptt_press(void)
{
    if (!s_have_mic)
        return;
    spk_stop();   /* talking over the agent cuts it off */
    s_ptt = true;
    STATE_LOCK();
    g_state.voice = VOICE_LISTENING;
    STATE_UNLOCK();
}

void mic_ptt_release(void)
{
    s_ptt = false;   /* task sends voice.end */
}

bool mic_ptt_held(void)
{
    return s_ptt;
}

void mic_ptt_task(void *arg)
{
    int16_t *buf = heap_caps_malloc(CHUNK_SAMPLES * sizeof(int16_t),
                                    MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!buf) {
        ESP_LOGE(TAG, "no mic buf");
        vTaskDelete(NULL);
    }
    while (1) {
        while (!s_ptt || !net_ws_connected())
            vTaskDelay(pdMS_TO_TICKS(20));

        char start[112];
        STATE_LOCK();
        bool interview = g_state.mode == MODE_INTERVIEW ||
                         (g_state.run_active && g_state.n_societies == 0);
        int agent = g_state.iv_agent.id;
        STATE_UNLOCK();
        if (interview && agent >= 0)
            snprintf(start, sizeof(start), "{\"type\":\"voice.start\",\"rate\":16000,"
                     "\"purpose\":\"interview\",\"agent\":%d}", agent);
        else if (interview)
            snprintf(start, sizeof(start),
                     "{\"type\":\"voice.start\",\"rate\":16000,\"purpose\":\"pick\"}");
        else
            snprintf(start, sizeof(start), "{\"type\":\"voice.start\",\"rate\":16000}");
        if (net_ws_send_text(start) < 0) {
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }
        int64_t t0 = esp_timer_get_time();
        bool cancelled = false;
        int total = 0;
        ESP_LOGI(TAG, "voice.start");

        while (s_ptt && net_ws_connected()) {
            size_t bytes = 0;
            /* read the i2s channel directly - esp_codec_dev_read() returns a
             * status (0 = OK), not a byte count */
            if (i2s_channel_read(s_rx, buf, CHUNK_SAMPLES * 2, &bytes,
                                 pdMS_TO_TICKS(500)) != ESP_OK || bytes == 0) {
                ESP_LOGW(TAG, "mic read got %d", (int)bytes);
                break;
            }
            total += bytes;
            int level;
            process(buf, bytes / 2, &level);
            STATE_LOCK();
            g_state.mic_level = level;
            STATE_UNLOCK();
            if (net_ws_send_audio(buf, bytes) < 0)
                break;
            if (esp_timer_get_time() - t0 > (int64_t)MAX_CLIP_MS * 1000) {
                cancelled = true;   /* server-side cap reached */
                break;
            }
        }

        net_ws_send_text(cancelled ? "{\"type\":\"voice.cancel\"}"
                                   : "{\"type\":\"voice.end\"}");
        s_ptt = false;
        STATE_LOCK();
        g_state.mic_level = 0;
        if (g_state.voice == VOICE_LISTENING)
            g_state.voice = VOICE_TRANSCRIBING;
        if (interview && !cancelled)
            g_state.iv_status = IV_THINKING;
        STATE_UNLOCK();
        ESP_LOGI(TAG, "voice.%s (%d bytes, %.1fs)", cancelled ? "cancel" : "end",
                 total, total / 32000.0);
    }
}

/* ---------------- speaker ---------------- */

void spk_begin(int bytes)
{
    s_spk_gen++;
    s_spk_done = false;
    s_spk_wr = 0;
    ESP_LOGI(TAG, "reply clip: %d bytes", bytes);
}

void spk_feed(const void *pcm, size_t len)
{
    if (!s_spk || s_spk_done)
        return;
    size_t wr = s_spk_wr;
    if (len > SPK_MAX - wr)
        len = SPK_MAX - wr;
    memcpy(s_spk + wr, pcm, len);
    s_spk_wr = wr + len;
}

void spk_end(void)
{
    s_spk_done = true;
}

void spk_stop(void)
{
    s_spk_gen++;
    s_spk_done = true;
    s_spk_wr = 0;
}

void spk_task(void *arg)
{
    while (1) {
        while (!s_have_mic || !s_spk || s_spk_wr == 0)
            vTaskDelay(pdMS_TO_TICKS(20));
        uint32_t gen = s_spk_gen;
        while (gen == s_spk_gen && !s_spk_done && s_spk_wr < SPK_PREBUF)
            vTaskDelay(pdMS_TO_TICKS(10));
        if (gen != s_spk_gen)
            continue;

        gpio_set_level(PIN_AMP_EN, 0);
        size_t rd = 0;
        while (gen == s_spk_gen) {
            size_t wr = s_spk_wr;
            if (rd >= wr) {
                if (s_spk_done)
                    break;
                vTaskDelay(pdMS_TO_TICKS(10));   /* network underrun */
                continue;
            }
            size_t n = wr - rd, w = 0;
            if (n > 2048)
                n = 2048;
            i2s_channel_write(s_tx, s_spk + rd, n, &w, pdMS_TO_TICKS(200));
            rd += w;
        }
        vTaskDelay(pdMS_TO_TICKS(150));   /* let the DMA drain */
        gpio_set_level(PIN_AMP_EN, 1);
        ESP_LOGI(TAG, "reply played (%d bytes)", (int)rd);
        if (gen == s_spk_gen) {
            s_spk_wr = 0;
            STATE_LOCK();
            if (g_state.iv_status == IV_SPEAKING)
                g_state.iv_status = IV_IDLE;
            STATE_UNLOCK();
        }
    }
}
