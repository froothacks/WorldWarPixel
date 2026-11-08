#pragma once

#include <stdint.h>
#include <stdbool.h>

#define LCD_W 320
#define LCD_H 480

/* RGB565, byte-swapped for SPI transport (panel expects MSB-first) */
#define RGB565(r, g, b) \
    (__builtin_bswap16((uint16_t)((((r) & 0xF8) << 8) | (((g) & 0xFC) << 3) | ((b) >> 3))))

void gfx_init(void);
void gfx_begin_frame(void);          /* swap buffer + copy background */
uint16_t *gfx_fb(void);              /* current front buffer (ready to present) */
void gfx_bg_target(void);            /* render subsequent primitives into the background */
void gfx_fb_target(void);            /* render subsequent primitives into the frame buffer */

void gfx_set_clip(int x0, int y0, int x1, int y1);
void gfx_reset_clip(void);

void gfx_fill_rect(int x, int y, int w, int h, uint16_t c);
void gfx_draw_rect_outline(int x, int y, int w, int h, uint16_t c);
void gfx_draw_line(int x0, int y0, int x1, int y1, uint16_t c);
void gfx_draw_thick_line(int x0, int y0, int x1, int y1, int t, uint16_t c);
void gfx_fill_circle(int cx, int cy, int r, uint16_t c);
void gfx_draw_circle_outline(int cx, int cy, int r, uint16_t c);
void gfx_fill_ellipse(int cx, int cy, int rx, int ry, uint16_t c);
void gfx_fill_ellipse_lower(int cx, int cy, int rx, int ry, uint16_t c);

int gfx_text_width(const char *s, int scale);
void gfx_draw_text(int x, int y, const char *s, int scale, uint16_t c);
void gfx_draw_text_centered(int cx, int y, const char *s, int scale, uint16_t c);
