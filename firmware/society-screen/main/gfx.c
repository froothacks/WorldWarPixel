#include "gfx.h"
#include "font.h"

#include <string.h>
#include <math.h>
#include "esp_heap_caps.h"

static uint16_t *fbs[2];
static uint16_t *bg;
static uint16_t *target;
static int cur;

/* clipping window */
static int cl_x0, cl_y0, cl_x1, cl_y1;

static inline void px(int x, int y, uint16_t c)
{
    if (x < cl_x0 || x >= cl_x1 || y < cl_y0 || y >= cl_y1)
        return;
    target[y * LCD_W + x] = c;
}

void gfx_init(void)
{
    fbs[0] = heap_caps_malloc(LCD_W * LCD_H * sizeof(uint16_t), MALLOC_CAP_SPIRAM);
    fbs[1] = heap_caps_malloc(LCD_W * LCD_H * sizeof(uint16_t), MALLOC_CAP_SPIRAM);
    bg = heap_caps_malloc(LCD_W * LCD_H * sizeof(uint16_t), MALLOC_CAP_SPIRAM);
    assert(fbs[0] && fbs[1] && bg);
    memset(fbs[0], 0, LCD_W * LCD_H * sizeof(uint16_t));
    memset(fbs[1], 0, LCD_W * LCD_H * sizeof(uint16_t));
    memset(bg, 0, LCD_W * LCD_H * sizeof(uint16_t));
    cur = 0;
    target = fbs[0];
    gfx_reset_clip();
}

void gfx_begin_frame(void)
{
    cur ^= 1;
    target = fbs[cur];
    memcpy(target, bg, LCD_W * LCD_H * sizeof(uint16_t));
}

uint16_t *gfx_fb(void)
{
    return fbs[cur];
}

void gfx_bg_target(void)
{
    target = bg;
}

void gfx_fb_target(void)
{
    target = fbs[cur];
}

void gfx_set_clip(int x0, int y0, int x1, int y1)
{
    cl_x0 = x0; cl_y0 = y0; cl_x1 = x1; cl_y1 = y1;
}

void gfx_reset_clip(void)
{
    cl_x0 = 0; cl_y0 = 0; cl_x1 = LCD_W; cl_y1 = LCD_H;
}

void gfx_fill_rect(int x, int y, int w, int h, uint16_t c)
{
    int x1 = x + w, y1 = y + h;
    if (x < cl_x0) x = cl_x0;
    if (y < cl_y0) y = cl_y0;
    if (x1 > cl_x1) x1 = cl_x1;
    if (y1 > cl_y1) y1 = cl_y1;
    for (int yy = y; yy < y1; yy++) {
        uint16_t *row = target + yy * LCD_W;
        for (int xx = x; xx < x1; xx++)
            row[xx] = c;
    }
}

void gfx_draw_rect_outline(int x, int y, int w, int h, uint16_t c)
{
    gfx_fill_rect(x, y, w, 1, c);
    gfx_fill_rect(x, y + h - 1, w, 1, c);
    gfx_fill_rect(x, y, 1, h, c);
    gfx_fill_rect(x + w - 1, y, 1, h, c);
}

void gfx_draw_line(int x0, int y0, int x1, int y1, uint16_t c)
{
    int dx = x1 - x0, dy = y1 - y0;
    int sx = dx > 0 ? 1 : -1, sy = dy > 0 ? 1 : -1;
    dx = dx > 0 ? dx : -dx;
    dy = dy > 0 ? dy : -dy;
    int err = dx - dy;
    for (;;) {
        px(x0, y0, c);
        if (x0 == x1 && y0 == y1)
            break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx)  { err += dx; y0 += sy; }
    }
}

void gfx_draw_thick_line(int x0, int y0, int x1, int y1, int t, uint16_t c)
{
    for (int i = 0; i < t; i++)
        gfx_draw_line(x0 + i, y0, x1 + i, y1, c);
}

void gfx_fill_circle(int cx, int cy, int r, uint16_t c)
{
    for (int dy = -r; dy <= r; dy++) {
        int dxmax = (int)(r * r) - dy * dy;
        if (dxmax < 0)
            continue;
        int dx = (int)sqrtf((float)dxmax);
        int x0 = cx - dx, x1 = cx + dx, yy = cy + dy;
        if (yy < cl_y0 || yy >= cl_y1)
            continue;
        if (x0 < cl_x0) x0 = cl_x0;
        if (x1 >= cl_x1) x1 = cl_x1 - 1;
        uint16_t *row = target + yy * LCD_W;
        for (int xx = x0; xx <= x1; xx++)
            row[xx] = c;
    }
}

void gfx_draw_circle_outline(int cx, int cy, int r, uint16_t c)
{
    for (int dy = -r; dy <= r; dy++) {
        int dxmax = r * r - dy * dy;
        if (dxmax < 0)
            continue;
        int dx = (int)sqrtf((float)dxmax);
        px(cx - dx, cy + dy, c);
        px(cx + dx, cy + dy, c);
        px(cx + dy, cy - dx, c);
        px(cx + dy, cy + dx, c);
    }
}

void gfx_fill_ellipse(int cx, int cy, int rx, int ry, uint16_t c)
{
    int64_t rx2 = (int64_t)rx * rx, ry2 = (int64_t)ry * ry;
    for (int dy = -ry; dy <= ry; dy++) {
        int64_t v = ry2 - (int64_t)dy * dy;
        if (v < 0)
            continue;
        int dx = (int)sqrtf((float)(rx2 * v / ry2));
        int x0 = cx - dx, x1 = cx + dx, yy = cy + dy;
        if (yy < cl_y0 || yy >= cl_y1)
            continue;
        if (x0 < cl_x0) x0 = cl_x0;
        if (x1 >= cl_x1) x1 = cl_x1 - 1;
        uint16_t *row = target + yy * LCD_W;
        for (int xx = x0; xx <= x1; xx++)
            row[xx] = c;
    }
}

void gfx_fill_ellipse_lower(int cx, int cy, int rx, int ry, uint16_t c)
{
    int64_t rx2 = (int64_t)rx * rx, ry2 = (int64_t)ry * ry;
    for (int dy = 0; dy <= ry; dy++) {
        int64_t v = ry2 - (int64_t)dy * dy;
        if (v < 0)
            continue;
        int dx = (int)sqrtf((float)(rx2 * v / ry2));
        int x0 = cx - dx, x1 = cx + dx, yy = cy + dy;
        if (yy < cl_y0 || yy >= cl_y1)
            continue;
        if (x0 < cl_x0) x0 = cl_x0;
        if (x1 >= cl_x1) x1 = cl_x1 - 1;
        uint16_t *row = target + yy * LCD_W;
        for (int xx = x0; xx <= x1; xx++)
            row[xx] = c;
    }
}

int gfx_text_width(const char *s, int scale)
{
    return (int)strlen(s) * (FONT_W + 1) * scale;
}

void gfx_draw_text(int x, int y, const char *s, int scale, uint16_t c)
{
    int cx = x;
    for (const char *p = s; *p; p++) {
        if (*p >= ' ' && *p <= '~') {
            const uint8_t *glyph = font5x7[*p - ' '];
            for (int col = 0; col < FONT_W; col++) {
                uint8_t bits = glyph[col];
                for (int row = 0; row < FONT_H; row++) {
                    if (bits & (1 << row)) {
                        if (scale == 1) {
                            px(cx + col, y + row, c);
                        } else {
                            gfx_fill_rect(cx + col * scale, y + row * scale, scale, scale, c);
                        }
                    }
                }
            }
        }
        cx += (FONT_W + 1) * scale;
    }
}

void gfx_draw_text_centered(int cx, int y, const char *s, int scale, uint16_t c)
{
    gfx_draw_text(cx - gfx_text_width(s, scale) / 2, y, s, scale, c);
}
