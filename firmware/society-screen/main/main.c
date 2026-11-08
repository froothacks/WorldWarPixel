#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_mac.h"

#include "hw.h"
#include "gfx.h"
#include "state.h"
#include "net.h"
#include "mic.h"
#include "ui.h"

app_state_t g_state;
SemaphoreHandle_t g_state_mtx;

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    g_state_mtx = xSemaphoreCreateMutex();
    g_state.conn = CONN_OFFLINE;
    g_state.voice = VOICE_IDLE;
    g_state.preview_society = -1;
    g_state.iv_agent.id = -1;
    net_load_mode();

    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    snprintf(g_state.mac, sizeof(g_state.mac), "%02X:%02X:%02X:%02X:%02X:%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    printf("\nWorldWarPixel society screen\n");
    printf("LCDWIKI ES3C35P / ST77922 / ST7123 / ES8311\n");
    printf("MAC %s\n\n", g_state.mac);

    gfx_init();
    hw_backlight_init();
    hw_display_init();
    hw_touch_init();
    hw_ws2812_init();

    hw_backlight_set(90);
    hw_ws2812_set(0, 0, 8);

    mic_init();                       /* tolerant: voice disabled on failure */
    xTaskCreate(mic_ptt_task, "mic", 6144, NULL, 5, NULL);
    xTaskCreate(spk_task, "spk", 4096, NULL, 6, NULL);
    xTaskCreate(ui_task, "ui", 8192, NULL, 5, NULL);

    net_start();                      /* wifi + ws + canvas poller */
}
