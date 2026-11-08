#include "hw.h"
#include "gfx.h"

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_rom_sys.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_vendor.h"
#include "driver/spi_master.h"
#include "driver/i2c_master.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "led_strip.h"
#include "nvs_flash.h"
#include "nvs.h"

#include "esp_lcd_st77922.h"
#include "esp_lcd_touch.h"
#include "esp_lcd_touch_st7123.h"

static const char *TAG = "hw";

/* LCDWIKI ES3C35P pin map (verified against community hardware reference) */
#define PIN_LCD_SCLK  12
#define PIN_LCD_D0    11
#define PIN_LCD_D1    13
#define PIN_LCD_D2    14
#define PIN_LCD_D3    9
#define PIN_LCD_CS    10
#define PIN_BL        41

#define PIN_TOUCH_SDA 38
#define PIN_TOUCH_SCL 39
#define PIN_TOUCH_RST 48
#define PIN_TOUCH_INT 47

#define PIN_WS2812    40

#define LCD_HOST SPI2_HOST

static esp_lcd_panel_handle_t panel;
static esp_lcd_touch_handle_t touch;
static led_strip_handle_t led;
static i2c_master_bus_handle_t i2c_bus;

static SemaphoreHandle_t flush_done;
static volatile bool flush_pending;

static bool IRAM_ATTR on_color_trans_done(esp_lcd_panel_io_handle_t io,
                                          esp_lcd_panel_io_event_data_t *edata,
                                          void *ctx)
{
    BaseType_t woken = pdFALSE;
    xSemaphoreGiveFromISR(flush_done, &woken);
    return woken == pdTRUE;
}

/* Vendor init sequence for the ES3C35P's ST77922 panel (320x480).
 * This is the 56-command table from the board's factory firmware / LCDwiki's
 * Arduino examples - the component's built-in default is for a different
 * 532x300 panel and leaves this one black. Tail mirrors the vendor table:
 * entry 58 = SLPOUT/120ms, last entry = TEON/20ms. */
static const st77922_lcd_init_cmd_t es3c35p_init_cmds[] = {
    {0xF1, (uint8_t[]){0x00}, 1, 0},
    {0x60, (uint8_t[]){0x00, 0x00, 0x00}, 3, 0},
    {0x65, (uint8_t[]){0x80}, 1, 0},
    {0x79, (uint8_t[]){0x06}, 1, 0},
    {0x7B, (uint8_t[]){0x00, 0x08, 0x08}, 3, 0},
    {0x80, (uint8_t[]){0x55, 0x62, 0x2F, 0x17, 0xF0, 0x52, 0x70, 0xD2, 0x52, 0x62, 0xEA}, 11, 0},
    {0x81, (uint8_t[]){0x26, 0x52, 0x72, 0x27}, 4, 0},
    {0x84, (uint8_t[]){0x92, 0x25}, 2, 0},
    {0x87, (uint8_t[]){0x10, 0x10, 0x58, 0x00, 0x02, 0x3A}, 6, 0},
    {0x88, (uint8_t[]){0x00, 0x00, 0x2C, 0x10, 0x04, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x06}, 15, 0},
    {0x89, (uint8_t[]){0x00, 0x00, 0x00}, 3, 0},
    {0x8A, (uint8_t[]){0x13, 0x00, 0x2C, 0x00, 0x00, 0x2C, 0x10, 0x10, 0x00, 0x3E, 0x19}, 11, 0},
    {0x8B, (uint8_t[]){0x15, 0xB1, 0xB1, 0x44, 0x96, 0x2C, 0x10, 0x97, 0x8E}, 9, 0},
    {0x8C, (uint8_t[]){0x1D, 0xB1, 0xB1, 0x44, 0x96, 0x2C, 0x10, 0x50, 0x0F, 0x01, 0xC5, 0x12, 0x09}, 13, 0},
    {0x8D, (uint8_t[]){0x0C}, 1, 0},
    {0x8E, (uint8_t[]){0x33, 0x01, 0x0C, 0x13, 0x01, 0x01}, 6, 0},
    {0xB3, (uint8_t[]){0x00, 0x30}, 2, 0},
    {0xF1, (uint8_t[]){0x00}, 1, 0},
    {0x71, (uint8_t[]){0xD0}, 1, 0},
    {0x66, (uint8_t[]){0x02, 0x3F}, 2, 0},
    {0xBE, (uint8_t[]){0x26, 0x00, 0x9D}, 3, 0},
    {0x70, (uint8_t[]){0x01, 0xA6, 0x11, 0x40, 0xE0, 0x00, 0x11, 0x60, 0x11, 0x00, 0x00, 0x1A}, 12, 0},
    {0x90, (uint8_t[]){0x04, 0x04, 0x55, 0x74, 0x00, 0x40, 0x43, 0x2D, 0x2D}, 9, 0},
    {0x91, (uint8_t[]){0x04, 0x04, 0x55, 0x75, 0x00, 0x40, 0x42, 0x2D, 0x2D}, 9, 0},
    {0x92, (uint8_t[]){0x04, 0x44, 0x55, 0xC0, 0x06, 0x00, 0x07, 0x05, 0x90, 0x2D}, 10, 0},
    {0x93, (uint8_t[]){0x04, 0x43, 0x11, 0x00, 0x00, 0x00, 0x00, 0x05, 0x90, 0x2D}, 10, 0},
    {0x94, (uint8_t[]){0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, 6, 0},
    {0x95, (uint8_t[]){0x96, 0x16, 0x00, 0x00, 0xFF}, 5, 0},
    {0x96, (uint8_t[]){0x44, 0x53, 0x03, 0x12, 0x23, 0x24, 0x06, 0x05, 0x9A, 0x2D, 0x00, 0x44}, 12, 0},
    {0x97, (uint8_t[]){0x44, 0x53, 0x47, 0x56, 0x20, 0x20, 0x02, 0x01, 0x9A, 0x2D, 0x00, 0x44}, 12, 0},
    {0xBA, (uint8_t[]){0x55, 0x9A, 0x2D, 0x9A, 0x2D}, 5, 0},
    {0x9A, (uint8_t[]){0x40, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00}, 7, 0},
    {0x9B, (uint8_t[]){0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00}, 7, 0},
    {0x9C, (uint8_t[]){0x5C, 0x12, 0x00, 0x00, 0x10, 0x12, 0x00, 0x00, 0x10, 0x02, 0x00, 0x00, 0x00}, 13, 0},
    {0x9D, (uint8_t[]){0x8A, 0x51, 0x00, 0x00, 0x00, 0x80, 0x1E, 0x01}, 8, 0},
    {0x9E, (uint8_t[]){0x51, 0x00, 0x00, 0x00, 0x80, 0x1E, 0x01}, 7, 0},
    {0xB4, (uint8_t[]){0x1D, 0x1C, 0x1E, 0x0B, 0x14, 0x02, 0x13, 0x09, 0x1E, 0x00, 0x1E, 0x10}, 12, 0},
    {0xB5, (uint8_t[]){0x1D, 0x1C, 0x1E, 0x0A, 0x15, 0x03, 0x11, 0x08, 0x1E, 0x01, 0x1E, 0x12}, 12, 0},
    {0xB6, (uint8_t[]){0x77, 0x77, 0x00, 0x0A, 0xFF, 0x0A, 0xFF}, 7, 0},
    {0x86, (uint8_t[]){0xC6, 0x04, 0xB1, 0x02, 0x58, 0x12, 0x58, 0x0C, 0x13, 0x01, 0xA5, 0x00, 0xA5, 0xA5}, 14, 0},
    {0xB7, (uint8_t[]){0x07, 0x0A, 0x0E, 0x06, 0x05, 0x03, 0x2B, 0x03, 0x03, 0x42, 0x07, 0x10, 0x10, 0x2E, 0x3F, 0x0D}, 16, 0},
    {0xB8, (uint8_t[]){0x07, 0x0A, 0x0D, 0x05, 0x05, 0x02, 0x2B, 0x02, 0x03, 0x42, 0x06, 0x10, 0x0F, 0x2E, 0x3F, 0x0D}, 16, 0},
    {0xB9, (uint8_t[]){0x23, 0x23}, 2, 0},
    {0xBF, (uint8_t[]){0x10, 0x14, 0x14, 0x0B, 0x0B, 0x0B}, 6, 0},
    {0xF2, (uint8_t[]){0x00}, 1, 0},
    {0x73, (uint8_t[]){0x04, 0xDA, 0x12, 0x54, 0x47}, 5, 0},
    {0x77, (uint8_t[]){0x6B, 0x5B, 0xFD, 0xC3, 0xC5}, 5, 0},
    {0x7A, (uint8_t[]){0x15, 0x27}, 2, 0},
    {0x7B, (uint8_t[]){0x04, 0x57}, 2, 0},
    {0x7E, (uint8_t[]){0x01, 0x0E}, 2, 0},
    {0xBF, (uint8_t[]){0x36}, 1, 0},
    {0xE3, (uint8_t[]){0x40, 0x40}, 2, 0},
    {0xF0, (uint8_t[]){0x00}, 1, 0},
    {0xD0, (uint8_t[]){0x00}, 1, 0},
    {0x2A, (uint8_t[]){0x00, 0x00, 0x01, 0x3F}, 4, 0},
    {0x2B, (uint8_t[]){0x00, 0x00, 0x01, 0xDF}, 4, 0},
    {0x11, NULL, 0, 200},                 /* SLPOUT + settle (vendor pairs 120ms) */
    {0x35, (uint8_t[]){0x00}, 1, 20},     /* TEON */
};

void hw_display_init(void)
{
    flush_done = xSemaphoreCreateBinary();
    flush_pending = false;

    ESP_LOGI(TAG, "init QSPI bus");
    const spi_bus_config_t buscfg = ST77922_PANEL_BUS_QSPI_CONFIG(
        PIN_LCD_SCLK, PIN_LCD_D0, PIN_LCD_D1, PIN_LCD_D2, PIN_LCD_D3,
        LCD_W * LCD_H * sizeof(uint16_t) + 64);
    ESP_ERROR_CHECK(spi_bus_initialize(LCD_HOST, &buscfg, SPI_DMA_CH_AUTO));

    ESP_LOGI(TAG, "install panel IO");
    esp_lcd_panel_io_handle_t io = NULL;
    const esp_lcd_panel_io_spi_config_t io_config =
        ST77922_PANEL_IO_QSPI_CONFIG(PIN_LCD_CS, on_color_trans_done, NULL);
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST,
                                             &io_config, &io));

    ESP_LOGI(TAG, "install st77922 panel driver");
    const st77922_vendor_config_t vendor_config = {
        .init_cmds = es3c35p_init_cmds,
        .init_cmds_size = sizeof(es3c35p_init_cmds) / sizeof(es3c35p_init_cmds[0]),
        .flags = {
            .use_qspi_interface = 1,
        },
    };
    const esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = -1,   /* panel reset is tied to CHIP_PU, not a GPIO */
        .rgb_ele_order = LCD_RGB_ELEMENT_ORDER_RGB,
        .bits_per_pixel = 16,
        .vendor_config = &vendor_config,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_st77922(io, &panel_config, &panel));

    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel, true));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel, true));
}

/* flush this many rows per SPI transaction. The framebuffer lives in PSRAM
 * which isn't DMA-capable, so the driver bounce-buffers each transaction in
 * internal RAM - a full-frame transfer can't allocate ~300KB there. 60 rows
 * needs a 38KB bounce buffer; waiting per strip keeps only one in flight. */
#define PRESENT_ROWS 60

void hw_present(uint16_t *fb)
{
    for (int y = 0; y < LCD_H; y += PRESENT_ROWS) {
        int rows = LCD_H - y;
        if (rows > PRESENT_ROWS)
            rows = PRESENT_ROWS;
        ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel, 0, y, LCD_W, y + rows,
                                                  fb + (size_t)y * LCD_W));
        xSemaphoreTake(flush_done, portMAX_DELAY);
    }
    flush_pending = false;
}

void hw_backlight_init(void)
{
    const ledc_timer_config_t timer = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .timer_num = LEDC_TIMER_0,
        .duty_resolution = LEDC_TIMER_8_BIT,
        .freq_hz = 20000,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer));
    const ledc_channel_config_t ch = {
        .gpio_num = PIN_BL,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = LEDC_CHANNEL_0,
        .timer_sel = LEDC_TIMER_0,
        .duty = 0,
        .hpoint = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&ch));
    hw_backlight_set(85);
}

void hw_backlight_set(int percent)
{
    if (percent < 0) percent = 0;
    if (percent > 100) percent = 100;
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, percent * 255 / 100);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
}

/* Bit-banged I2C bus recovery: a device stuck mid-transaction (the ST7123 is
 * known to hold SDA low while held in reset) releases the bus after 9 clocks. */
static void i2c_bus_recover(void)
{
    const gpio_config_t od = {
        .pin_bit_mask = BIT64(PIN_TOUCH_SDA) | BIT64(PIN_TOUCH_SCL),
        .mode = GPIO_MODE_INPUT_OUTPUT_OD,
        .pull_up_en = GPIO_PULLUP_ENABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&od));
    gpio_set_level(PIN_TOUCH_SDA, 1);
    gpio_set_level(PIN_TOUCH_SCL, 1);
    esp_rom_delay_us(10);
    for (int i = 0; i < 9 && gpio_get_level(PIN_TOUCH_SDA) == 0; i++) {
        gpio_set_level(PIN_TOUCH_SCL, 0);
        esp_rom_delay_us(10);
        gpio_set_level(PIN_TOUCH_SCL, 1);
        esp_rom_delay_us(10);
    }
    /* stop condition: SDA low -> high while SCL high */
    gpio_set_level(PIN_TOUCH_SDA, 0);
    esp_rom_delay_us(10);
    gpio_set_level(PIN_TOUCH_SDA, 1);
    esp_rom_delay_us(10);
}

void hw_touch_init(void)
{
    ESP_LOGI(TAG, "init touch I2C (st7123 @0x55)");

    /* hold the touch controller in reset while the bus is recovered/created */
    const gpio_config_t rst_conf = {
        .pin_bit_mask = BIT64(PIN_TOUCH_RST),
        .mode = GPIO_MODE_OUTPUT,
    };
    ESP_ERROR_CHECK(gpio_config(&rst_conf));
    gpio_set_level(PIN_TOUCH_RST, 0);

    i2c_bus_recover();

    i2c_master_bus_config_t i2c_bus_config = {
        .i2c_port = 0,
        .sda_io_num = PIN_TOUCH_SDA,
        .scl_io_num = PIN_TOUCH_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    ESP_ERROR_CHECK(i2c_new_master_bus(&i2c_bus_config, &i2c_bus));

    /* release reset; ST7123 needs up to ~20-30ms to initialize before it
     * answers I2C - give it plenty. NOTE: do NOT hand rst_gpio_num to the
     * driver below: its internal reset waits only 10ms after release, which
     * is inside the chip's init window and NACKs the first register read. */
    gpio_set_level(PIN_TOUCH_RST, 1);
    vTaskDelay(pdMS_TO_TICKS(300));

    /* diagnostic bus scan */
    for (int addr = 1; addr < 0x7F; addr++) {
        if (i2c_master_probe(i2c_bus, addr, 20) == ESP_OK)
            ESP_LOGI(TAG, "i2c device at 0x%02X", addr);
    }

    esp_lcd_panel_io_handle_t tp_io = NULL;
    esp_lcd_panel_io_i2c_config_t io_config = ESP_LCD_TOUCH_IO_I2C_ST7123_CONFIG();
    io_config.scl_speed_hz = 400000;
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_i2c_v2(i2c_bus, &io_config, &tp_io));

    const esp_lcd_touch_config_t tp_cfg = {
        .x_max = LCD_W,
        .y_max = LCD_H,
        .rst_gpio_num = GPIO_NUM_NC,   /* reset already done above with proper settle */
        .int_gpio_num = GPIO_NUM_NC,   /* polled mode */
        .flags = {
            .swap_xy = 0,
            .mirror_x = 0,
            .mirror_y = 0,
        },
    };

    esp_err_t err = ESP_FAIL;
    for (int attempt = 1; attempt <= 3 && err != ESP_OK; attempt++) {
        err = esp_lcd_touch_new_i2c_st7123(tp_io, &tp_cfg, &touch);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "touch init attempt %d failed", attempt);
            gpio_set_level(PIN_TOUCH_RST, 0);
            vTaskDelay(pdMS_TO_TICKS(20));
            gpio_set_level(PIN_TOUCH_RST, 1);
            vTaskDelay(pdMS_TO_TICKS(300));
        }
    }
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "TOUCH INIT FAILED - power-cycle the board (unplug/replug USB)");
        touch = NULL;
    }
}

bool hw_touch_poll(int *x, int *y)
{
    if (!touch)
        return false;
    ESP_ERROR_CHECK(esp_lcd_touch_read_data(touch));
    esp_lcd_touch_point_data_t points[1] = {0};
    uint8_t cnt = 0;
    esp_lcd_touch_get_data(touch, points, &cnt, 1);
    if (cnt > 0) {
        *x = points[0].x;
        *y = points[0].y;
        return true;
    }
    return false;
}

void hw_ws2812_init(void)
{
    const led_strip_config_t strip_config = {
        .strip_gpio_num = PIN_WS2812,
        .max_leds = 1,
        .led_model = LED_MODEL_WS2812,
    };
    const led_strip_rmt_config_t rmt_config = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10 * 1000 * 1000,
    };
    ESP_ERROR_CHECK(led_strip_new_rmt_device(&strip_config, &rmt_config, &led));
    ESP_ERROR_CHECK(led_strip_clear(led));
}

void hw_ws2812_set(int r, int g, int b)
{
    ESP_ERROR_CHECK(led_strip_set_pixel(led, 0, r, g, b));
    ESP_ERROR_CHECK(led_strip_refresh(led));
}

/* ES8311 codec and the touch controller share this I2C bus. */
i2c_master_bus_handle_t hw_i2c_bus(void)
{
    return i2c_bus;
}
