#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "driver/i2c_master.h"

void hw_display_init(void);
void hw_present(uint16_t *fb);       /* wait for previous DMA, then start new frame */

void hw_backlight_init(void);
void hw_backlight_set(int percent);

void hw_touch_init(void);
bool hw_touch_poll(int *x, int *y);   /* current touch state + coordinates */

void hw_ws2812_init(void);
void hw_ws2812_set(int r, int g, int b);

i2c_master_bus_handle_t hw_i2c_bus(void);   /* shared bus: touch + ES8311 codec */
