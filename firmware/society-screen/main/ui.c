#include "ui.h"
#include "hw.h"
#include "gfx.h"
#include "font.h"
#include "mic.h"
#include "net.h"
#include "state.h"
#include "secrets.h"

#include <string.h>
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "ui";

/* palette */
#define C_BG      RGB565(16, 18, 24)
#define C_PANEL   RGB565(30, 33, 42)
#define C_TEXT    RGB565(235, 235, 240)
#define C_DIM     RGB565(130, 135, 145)
#define C_LINE    RGB565(55, 60, 72)
#define C_RED     RGB565(220, 60, 60)
#define C_GREEN   RGB565(60, 180, 90)
#define C_YELLOW  RGB565(220, 180, 40)

#define BTN_X 16
#define BTN_Y 392
#define BTN_W (LCD_W - 32)
#define BTN_H 76

#define TAB_Y  2
#define TAB_H  20
#define TAB_W  74
#define TAB1_X (LCD_W - 2 * TAB_W - 10)
#define TAB2_X (LCD_W - TAB_W - 4)

#define IV_SESSION_Y 92
#define LIST_Y       112
#define ROW_H        26
#define LIST_ROWS    9
#define PAGER_Y      350
#define CONVO_Y      110
#define CONVO_BOTTOM 384

static uint16_t hex_to_565(const char *s)
{
    if (!s || s[0] != '#')
        return C_DIM;
    unsigned v = (unsigned)strtoul(s + 1, NULL, 16);
    return RGB565((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF);
}

static uint16_t rgb_to_565(uint32_t v)
{
    return RGB565((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF);
}

/* ellipsise into a fixed pixel width at given scale */
static void ellipsize(char *s, int max_px, int scale)
{
    int max_chars = max_px / ((FONT_W + 1) * scale);
    int n = (int)strlen(s);
    if (n <= max_chars)
        return;
    if (max_chars > 3) {
        strcpy(s + max_chars - 3, "...");
    } else {
        s[max_chars < 0 ? 0 : max_chars] = 0;
    }
}

/* crude word-wrap: draw text in a box, returns lines used */
static int draw_wrapped(int x, int y, int w, const char *s, int scale, uint16_t c, int max_lines)
{
    char tmp[512];
    strncpy(tmp, s, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = 0;
    int lines = 0, cx = x;
    int char_w = (FONT_W + 1) * scale;
    for (char *p = strtok(tmp, " "); p; p = strtok(NULL, " ")) {
        int wp = gfx_text_width(p, scale) + char_w;
        if (cx + wp - char_w > x + w) {
            if (++lines >= max_lines)
                return lines;
            cx = x;
            y += (FONT_H + 3) * scale;
        }
        gfx_draw_text(cx, y, p, scale, c);
        cx += wp;
    }
    return lines + 1;
}

static void draw_status(const app_state_t *st)
{
    const char *txt = "?";
    uint16_t col = C_DIM;
    switch (st->conn) {
    case CONN_OFFLINE:    txt = "offline - retrying";        col = C_RED; break;
    case CONN_WIFI:       txt = "not connected";             col = C_YELLOW; break;
    case CONN_WS:         txt = "connected";                 col = C_GREEN; break;
    case CONN_NOT_PAIRED: txt = "NOT PAIRED";                col = C_RED; break;
    }
    gfx_draw_text(8, 5, txt, 1, col);
}

static void draw_tabs(const app_state_t *st)
{
    const char *lbl[2] = {"SOCIETY", "INTERVIEW"};
    const int xs[2] = {TAB1_X, TAB2_X};
    for (int i = 0; i < 2; i++) {
        bool on = (int)st->mode == i;
        gfx_fill_rect(xs[i], TAB_Y, TAB_W, TAB_H, on ? C_TEXT : C_PANEL);
        gfx_draw_rect_outline(xs[i], TAB_Y, TAB_W, TAB_H, on ? C_TEXT : C_LINE);
        gfx_draw_text_centered(xs[i] + TAB_W / 2, TAB_Y + 7, lbl[i], 1, on ? C_BG : C_DIM);
    }
}

static void draw_header(const app_state_t *st)
{
    uint16_t chip = rgb_to_565(st->society_color);
    gfx_fill_rect(8, 28, 56, 56, chip);
    gfx_draw_rect_outline(8, 28, 56, 56, C_LINE);

    char name[32];
    strncpy(name, st->society_name[0] ? st->society_name : "Society", sizeof(name) - 1);
    name[sizeof(name) - 1] = 0;
    ellipsize(name, 240, 3);
    gfx_draw_text(76, 34, name, 3, C_TEXT);

    char sub[48];
    snprintf(sub, sizeof(sub), "%d members", st->members);
    gfx_draw_text(76, 62, sub, 1, C_DIM);

    char mig[32] = "";
    if (st->joined || st->left)
        snprintf(mig, sizeof(mig), "+%d / -%d", st->joined, st->left);
    gfx_draw_text(76, 74, mig, 1, C_GREEN);
}

static void draw_society_strip(const app_state_t *st)
{
    int seg_w = LCD_W / MAX_SOCIETIES;
    for (int i = 0; i < st->n_societies && i < MAX_SOCIETIES; i++) {
        const society_brief_t *s = &st->societies[i];
        int x = i * seg_w;
        bool own = (s->id == st->society_id);
        uint16_t col = own ? C_TEXT : C_DIM;
        char lbl[40];
        snprintf(lbl, sizeof(lbl), "%d %s (%d)", s->id + 1, s->name, s->members);
        ellipsize(lbl, seg_w - 12, 1);
        gfx_draw_text(x + 8, 102, lbl, 1, col);
        if (own)
            gfx_fill_rect(x + 8, 114, seg_w - 16, 2, rgb_to_565(st->society_color));
    }
    gfx_draw_line(0, 122, LCD_W, 122, C_LINE);
}

static void draw_session(const app_state_t *st, int y)
{
    char buf[96];
    if (st->run_active && st->run_id[0]) {
        if (st->run_pinned) {
            const run_info_t *ri = NULL;
            for (int i = 0; i < st->n_runs; i++)
                if (!strcmp(st->runs[i].id, st->run_id)) ri = &st->runs[i];
            if (ri)
                snprintf(buf, sizeof(buf), "SESSION %s  %d agents  %s",
                         st->run_id, ri->agents, ri->model);
            else
                snprintf(buf, sizeof(buf), "SESSION %s", st->run_id);
        } else {
            snprintf(buf, sizeof(buf), "SESSION latest (%s)", st->run_id);
        }
        ellipsize(buf, LCD_W - 16, 1);
        gfx_draw_text(8, y, buf, 1, C_TEXT);
    } else if (st->run_pinned) {
        gfx_draw_text(8, y, "SESSION ENDED - tap row to pick", 1, C_RED);
    } else if (st->conn == CONN_WS) {
        gfx_draw_text(8, y, "no live session", 1, C_DIM);
    }
}

static void draw_threads(const app_state_t *st)
{
    gfx_draw_text(8, 142, "MESSAGE BOARD", 1, C_DIM);
    if (st->n_threads == 0) {
        gfx_draw_text(8, 160, "no threads yet", 1, C_DIM);
        return;
    }
    int y = 160;
    for (int i = 0; i < st->n_threads; i++) {
        const thread_t *t = &st->threads[i];
        char title[64];
        strncpy(title, t->title, sizeof(title) - 1);
        title[sizeof(title) - 1] = 0;
        ellipsize(title, 304, 2);
        gfx_draw_text(8, y, title, 2, C_TEXT);
        char meta[64];
        snprintf(meta, sizeof(meta), "%s - %d replies",
                 t->author[0] ? t->author : "?", t->replies);
        ellipsize(meta, 304, 1);
        gfx_draw_text(8, y + 20, meta, 1, C_DIM);
        y += 38;
        if (i < st->n_threads - 1)
            gfx_draw_line(8, y - 6, LCD_W - 8, y - 6, C_LINE);
    }
}

static void draw_canvas(const app_state_t *st)
{
    /* thumbnail at (8, 286), max 160x100 shown as 2x2 decimation of 240x160 src */
    if (!st->canvas_idx || st->canvas_w <= 0 || st->n_palette == 0)
        return;
    int dw = 160, dh = 100;
    int sx_step = st->canvas_w / dw;   if (sx_step < 1) sx_step = 1;
    int sy_step = st->canvas_h / dh;   if (sy_step < 1) sy_step = 1;
    for (int y = 0; y < dh; y++) {
        int sy = y * sy_step;
        if (sy >= st->canvas_h) break;
        for (int x = 0; x < dw; x++) {
            int sx = x * sx_step;
            if (sx >= st->canvas_w) break;
            int idx = st->canvas_idx[sy * st->canvas_w + sx];
            if (idx >= st->n_palette) idx = 0;
            gfx_fill_rect(8 + x, 286 + y, 1, 1, hex_to_565(st->palette[idx]));
        }
    }
    gfx_draw_rect_outline(7, 285, 162, 102, C_LINE);
    gfx_draw_text(8, 274, "CANVAS", 1, C_DIM);
}

static void draw_murmur(const app_state_t *st)
{
    int x = 176, w = LCD_W - 176 - 8;
    if (!st->canvas_idx) { x = 8; w = LCD_W - 16; }
    gfx_draw_text(x, 274, "MURMUR", 1, C_DIM);
    if (st->has_murmur) {
        char buf[192];
        snprintf(buf, sizeof(buf), "\"%.188s\"", st->murmur);
        draw_wrapped(x, 288, w, buf, 1, C_YELLOW, 7);
    } else {
        gfx_draw_text(x, 288, "quiet...", 1, C_DIM);
    }
}

static void draw_button(const app_state_t *st, bool held, int64_t held_us)
{
    uint16_t fill = held ? C_RED : C_PANEL;
    gfx_fill_rect(BTN_X, BTN_Y, BTN_W, BTN_H, fill);
    gfx_draw_rect_outline(BTN_X, BTN_Y, BTN_W, BTN_H, held ? C_RED : C_LINE);

    const char *label;
    uint16_t lc = held ? C_TEXT : C_DIM;
    switch (st->voice) {
    case VOICE_TRANSCRIBING: label = "TRANSCRIBING..."; break;
    case VOICE_DONE:         label = "HEARD"; break;
    case VOICE_ERROR:        label = "VOICE ERROR"; break;
    default:                 label = held ? "LISTENING" : "HOLD TO TALK"; break;
    }
    gfx_draw_text_centered(LCD_W / 2, BTN_Y + 14, label, 2, lc);

    if (held) {
        /* mic level bar + elapsed */
        int lvl = st->mic_level;
        if (lvl > 100) lvl = 100;
        gfx_fill_rect(BTN_X + 12, BTN_Y + 44, (BTN_W - 24) * lvl / 100, 10, C_GREEN);
        gfx_draw_rect_outline(BTN_X + 12, BTN_Y + 44, BTN_W - 24, 10, C_LINE);
        char secs[16];
        snprintf(secs, sizeof(secs), "%0.1fs", held_us / 1000000.0);
        gfx_draw_text_centered(LCD_W / 2, BTN_Y + 60, secs, 1, C_TEXT);
    } else if (st->voice == VOICE_DONE && st->transcript[0]) {
        char buf[200];
        snprintf(buf, sizeof(buf), "Heard: %s", st->transcript);
        draw_wrapped(BTN_X + 8, BTN_Y + 40, BTN_W - 16, buf, 1, C_TEXT, 3);
    }
}

static void draw_toast(const app_state_t *st, int64_t now)
{
    if (now >= st->toast_until_us || !st->toast[0])
        return;
    char msg[96];
    strncpy(msg, st->toast, sizeof(msg) - 1);
    msg[sizeof(msg) - 1] = 0;
    ellipsize(msg, 288, 1);
    int w = gfx_text_width(msg, 1) + 16;
    int x = (LCD_W - w) / 2;
    gfx_fill_rect(x, 372, w, 16, C_PANEL);
    gfx_draw_rect_outline(x, 372, w, 16, C_LINE);
    gfx_draw_text(x + 8, 376, msg, 1, C_TEXT);
}

#define CONN_BOX_Y 180
#define CONN_BOX_H 100
#define CONN_BOX_W 260

/* shown whenever the socket is down: connecting only ever happens on this tap */
static void draw_connect(const app_state_t *st, int64_t now, int64_t flash_until)
{
    if (st->conn == CONN_OFFLINE) {
        gfx_draw_text_centered(LCD_W / 2, 220, "NO WIFI", 3, C_RED);
        gfx_draw_text_centered(LCD_W / 2, 256, "joining the network...", 1, C_DIM);
        return;
    }
    int phase, attempt;
    net_ws_phase(&phase, &attempt);
    bool lit = now < flash_until || phase == 2;
    int x = (LCD_W - CONN_BOX_W) / 2;
    gfx_fill_rect(x, CONN_BOX_Y, CONN_BOX_W, CONN_BOX_H, lit ? C_YELLOW : C_PANEL);
    gfx_draw_rect_outline(x, CONN_BOX_Y, CONN_BOX_W, CONN_BOX_H, C_YELLOW);
    gfx_draw_text_centered(LCD_W / 2, CONN_BOX_Y + 30, phase == 2 ? "CONNECTING..." : "CONNECT",
                           3, lit ? C_BG : C_YELLOW);
    const char *host = SERVER_URL;
    if (!strncmp(host, "https://", 8)) host += 8;
    else if (!strncmp(host, "http://", 7)) host += 7;
    gfx_draw_text_centered(LCD_W / 2, CONN_BOX_Y + 70, host, 1, lit ? C_BG : C_DIM);
    if (phase == 3)
        gfx_draw_text_centered(LCD_W / 2, CONN_BOX_Y + CONN_BOX_H + 14,
                               "could not reach the server - tap to retry", 1, C_RED);
}

#define SP_Y    60
#define SP_ROW  36
#define SP_ROWS (MAX_RUNS + 1)   /* row 0 = follow the newest session */

static void draw_session_picker(const app_state_t *st)
{
    gfx_draw_text(8, 30, "PICK A SESSION", 2, C_TEXT);
    if (st->n_runs == 0) {
        gfx_draw_text_centered(LCD_W / 2, 170, "NO LIVE SESSIONS", 2, C_YELLOW);
        draw_wrapped(24, 200, LCD_W - 48,
                     "Start a run with societies on the website; it shows up here.", 1, C_DIM, 3);
    }
    if (st->n_runs > 0) {
        bool cur = !st->run_pinned;
        gfx_fill_rect(8, SP_Y, LCD_W - 16, SP_ROW - 4, cur ? C_PANEL : C_BG);
        gfx_draw_rect_outline(8, SP_Y, LCD_W - 16, SP_ROW - 4, cur ? C_TEXT : C_LINE);
        gfx_draw_text(16, SP_Y + 4, "FOLLOW NEWEST", 2, C_TEXT);
        gfx_draw_text(16, SP_Y + 22, "switches to new sessions as they start", 1, C_DIM);
    }
    for (int r = 0; r < st->n_runs && r + 1 < SP_ROWS; r++) {
        const run_info_t *ri = &st->runs[r];
        int y = SP_Y + (r + 1) * SP_ROW;
        bool cur = st->run_pinned && !strcmp(ri->id, st->run_id);
        gfx_fill_rect(8, y, LCD_W - 16, SP_ROW - 4, cur ? C_PANEL : C_BG);
        gfx_draw_rect_outline(8, y, LCD_W - 16, SP_ROW - 4, cur ? C_TEXT : C_LINE);
        char l1[48];
        snprintf(l1, sizeof(l1), "%s  %d agents", ri->id, ri->agents);
        ellipsize(l1, LCD_W - 32, 2);
        gfx_draw_text(16, y + 4, l1, 2, C_TEXT);
        char l2[96];
        snprintf(l2, sizeof(l2), "%s - %s", ri->model, ri->soc_names);
        ellipsize(l2, LCD_W - 32, 1);
        gfx_draw_text(16, y + 22, l2, 1, C_DIM);
    }
    gfx_fill_rect(BTN_X, BTN_Y, BTN_W, BTN_H, C_PANEL);
    gfx_draw_rect_outline(BTN_X, BTN_Y, BTN_W, BTN_H, C_LINE);
    gfx_draw_text_centered(LCD_W / 2, BTN_Y + 30, "BACK", 2, C_DIM);
}

/* ---------------- interview mode ---------------- */

static int page_count(const app_state_t *st)
{
    return st->n_agents ? (st->n_agents + LIST_ROWS - 1) / LIST_ROWS : 1;
}

static void draw_iv_header(const app_state_t *st)
{
    if (st->iv_agent.id < 0) {
        gfx_draw_text(8, 34, "PICK SOMEONE", 3, C_TEXT);
        gfx_draw_text(8, 64, "tap a name, or hold the button and say it", 1, C_DIM);
        return;
    }
    gfx_fill_rect(8, 28, 56, 56, rgb_to_565(st->iv_agent.color));
    gfx_draw_rect_outline(8, 28, 56, 56, C_LINE);
    char name[28];
    snprintf(name, sizeof(name), "%s", st->iv_agent.name);
    ellipsize(name, 236, 2);
    gfx_draw_text(76, 32, name, 2, C_TEXT);
    char sub[80];
    const char *soc = NULL;
    for (int i = 0; i < st->n_societies; i++)
        if (st->societies[i].id == st->iv_agent.society) soc = st->societies[i].name;
    if (soc)
        snprintf(sub, sizeof(sub), "%s - %s", st->iv_agent.country, soc);
    else
        snprintf(sub, sizeof(sub), "%s", st->iv_agent.country);
    ellipsize(sub, 236, 1);
    gfx_draw_text(76, 54, sub, 1, C_DIM);
    gfx_draw_text(76, 70, "tap here to talk to someone else", 1, C_DIM);
}

static void draw_picker(const app_state_t *st, int page)
{
    if (st->n_agents == 0) {
        gfx_draw_text(8, LIST_Y + 8, st->run_active ? "loading people..." : "no live session",
                      2, C_DIM);
        return;
    }
    for (int r = 0; r < LIST_ROWS; r++) {
        int i = page * LIST_ROWS + r;
        if (i >= st->n_agents)
            break;
        const agent_info_t *a = &st->agents[i];
        int y = LIST_Y + r * ROW_H;
        gfx_fill_rect(8, y + 5, 14, 14, rgb_to_565(a->color));
        char name[28];
        snprintf(name, sizeof(name), "%s", a->name);
        ellipsize(name, 228, 2);
        gfx_draw_text(30, y + 5, name, 2, C_TEXT);
        if (a->society >= 0) {
            char sn[16];
            snprintf(sn, sizeof(sn), "S%d", a->society + 1);
            gfx_draw_text(LCD_W - 26, y + 9, sn, 1, C_DIM);
        }
        gfx_draw_line(8, y + ROW_H - 1, LCD_W - 8, y + ROW_H - 1, C_LINE);
    }
    int pages = page_count(st);
    if (pages > 1) {
        char pg[24];
        snprintf(pg, sizeof(pg), "%d / %d", page + 1, pages);
        gfx_draw_text(16, PAGER_Y + 6, "< PREV", 2, C_TEXT);
        gfx_draw_text_centered(LCD_W / 2, PAGER_Y + 10, pg, 1, C_DIM);
        gfx_draw_text(LCD_W - 16 - gfx_text_width("NEXT >", 2), PAGER_Y + 6, "NEXT >", 2, C_TEXT);
    }
}

static int est_lines(const char *s, int scale)
{
    int cpl = (LCD_W - 16) / ((FONT_W + 1) * scale);
    int n = (int)strlen(s);
    return (n * 115 / 100) / (cpl > 0 ? cpl : 1) + 1;   /* +15% for word-wrap slack */
}

static void draw_convo(const app_state_t *st)
{
    if (st->n_convo == 0) {
        draw_wrapped(8, CONVO_Y + 8, LCD_W - 16, "Hold the button and ask a question.", 2, C_DIM, 3);
        return;
    }
    const int avail = CONVO_BOTTOM - CONVO_Y;
    const convo_line_t *last = &st->convo[st->n_convo - 1];
    int scale_last = (!last->me && est_lines(last->text, 2) * 20 + 14 <= avail) ? 2 : 1;
    int start = st->n_convo, used = 0;
    while (start > 0) {
        const convo_line_t *l = &st->convo[start - 1];
        int sc = start == st->n_convo ? scale_last : 1;
        int h = 12 + est_lines(l->text, sc) * (FONT_H + 3) * sc + 6;
        if (used + h > avail && start != st->n_convo)
            break;
        used += h;
        start--;
    }
    int y = CONVO_Y;
    for (int i = start; i < st->n_convo; i++) {
        const convo_line_t *l = &st->convo[i];
        int sc = i == st->n_convo - 1 ? scale_last : 1;
        gfx_draw_text(8, y, l->me ? "YOU" : st->iv_agent.name, 1,
                      l->me ? C_DIM : rgb_to_565(st->iv_agent.color));
        y += 12;
        int max_lines = (CONVO_BOTTOM - y) / ((FONT_H + 3) * sc);
        if (max_lines < 1)
            break;
        int n = draw_wrapped(8, y, LCD_W - 16, l->text, sc, l->me ? C_DIM : C_TEXT, max_lines);
        y += n * (FONT_H + 3) * sc + 6;
    }
}

static void draw_iv_button(const app_state_t *st, bool held, int64_t held_us)
{
    gfx_fill_rect(BTN_X, BTN_Y, BTN_W, BTN_H, held ? C_RED : C_PANEL);
    gfx_draw_rect_outline(BTN_X, BTN_Y, BTN_W, BTN_H, held ? C_RED : C_LINE);
    char label[48];
    if (held)
        snprintf(label, sizeof(label), "LISTENING");
    else if (st->iv_status == IV_SPEAKING)
        snprintf(label, sizeof(label), "SPEAKING");
    else if (st->iv_status == IV_THINKING)
        snprintf(label, sizeof(label), "THINKING...");
    else if (st->iv_agent.id < 0)
        snprintf(label, sizeof(label), "HOLD + SAY A NAME");
    else {
        char first[28];
        snprintf(first, sizeof(first), "%s", st->iv_agent.name);
        char *sp = strchr(first, ' ');
        if (sp) *sp = 0;
        snprintf(label, sizeof(label), "HOLD TO ASK %s", first);
    }
    ellipsize(label, BTN_W - 16, 2);
    gfx_draw_text_centered(LCD_W / 2, BTN_Y + 14, label, 2, held ? C_TEXT : C_DIM);
    if (held) {
        int lvl = st->mic_level > 100 ? 100 : st->mic_level;
        gfx_fill_rect(BTN_X + 12, BTN_Y + 44, (BTN_W - 24) * lvl / 100, 10, C_GREEN);
        gfx_draw_rect_outline(BTN_X + 12, BTN_Y + 44, BTN_W - 24, 10, C_LINE);
        char secs[16];
        snprintf(secs, sizeof(secs), "%0.1fs", held_us / 1000000.0);
        gfx_draw_text_centered(LCD_W / 2, BTN_Y + 60, secs, 1, C_TEXT);
    } else if (st->iv_status == IV_SPEAKING) {
        gfx_draw_text_centered(LCD_W / 2, BTN_Y + 48, "hold to interrupt", 1, C_DIM);
    }
}

static void draw_not_paired(const app_state_t *st)
{
    gfx_draw_text_centered(LCD_W / 2, 160, "NOT PAIRED", 3, C_RED);
    gfx_draw_text_centered(LCD_W / 2, 200, "give the operator this MAC:", 1, C_DIM);
    gfx_draw_text_centered(LCD_W / 2, 218, st->mac, 2, C_TEXT);
    gfx_draw_text_centered(LCD_W / 2, 260, "tap to try again", 1, C_DIM);
}

void ui_task(void *arg)
{
    static app_state_t st;   /* too big for the task stack */
    bool held = false, prev_touched = false;
    int64_t held_since = 0;
    int64_t voice_done_at = 0;
    int64_t retry_flash_until = 0;
    bool picking = false;                /* session list on screen */
    conn_state_t prev_conn = CONN_OFFLINE;
    int page = 0;
    int64_t agents_asked = 0, thinking_since = 0;

    while (1) {
        STATE_LOCK();
        st = g_state;                  /* shallow copy incl. pointers */
        STATE_UNLOCK();
        if (st.conn == CONN_WS && prev_conn != CONN_WS)
            picking = true;            /* every new connection starts at the session list */
        prev_conn = st.conn;
        bool live = st.conn == CONN_WS && !picking;

        int tx, ty;
        bool touched = hw_touch_poll(&tx, &ty);
        bool in_btn = live && touched && tx >= BTN_X && tx < BTN_X + BTN_W &&
                      ty >= BTN_Y && ty < BTN_Y + BTN_H;

        if (in_btn && !held) {
            held = true;
            held_since = esp_timer_get_time();
            mic_ptt_press();
        } else if (held && (!touched || !in_btn)) {
            held = false;
            mic_ptt_release();
        }

        bool tap = touched && !prev_touched;
        bool interview = st.mode == MODE_INTERVIEW;
        int64_t now_us = esp_timer_get_time();

        /* mode tabs (top right) */
        if (tap && ty < TAB_Y + TAB_H + 4 && tx >= TAB1_X) {
            int mode = tx >= TAB2_X ? MODE_INTERVIEW : MODE_SOCIETY;
            if (mode != (int)st.mode) {
                net_set_mode(mode);
                spk_stop();
                page = 0;
            }
        }

        if (interview && live) {
            /* keep the list of people for the current world */
            if (st.conn == CONN_WS && st.run_active &&
                (strcmp(st.agents_run, st.run_id) || st.n_agents == 0) &&
                now_us - agents_asked > 4000000) {
                net_agents_list();
                agents_asked = now_us;
            }
            /* never get stuck on THINKING if the server drops the reply */
            if (st.iv_status != IV_THINKING)
                thinking_since = 0;
            else if (!thinking_since)
                thinking_since = now_us;
            else if (now_us - thinking_since > 60000000) {
                STATE_LOCK();
                g_state.iv_status = IV_IDLE;
                STATE_UNLOCK();
            }
            int pages = page_count(&st);
            if (page >= pages)
                page = pages - 1;
            if (tap && st.iv_agent.id >= 0 && ty >= 28 && ty < 88) {
                spk_stop();
                STATE_LOCK();
                g_state.iv_agent.id = -1;
                g_state.n_convo = 0;
                g_state.iv_status = IV_IDLE;
                STATE_UNLOCK();
            } else if (tap && st.iv_agent.id < 0 && ty >= LIST_Y &&
                       ty < LIST_Y + LIST_ROWS * ROW_H) {
                int i = page * LIST_ROWS + (ty - LIST_Y) / ROW_H;
                if (i < st.n_agents)
                    net_interview_select(st.agents[i].id);
            } else if (tap && st.iv_agent.id < 0 && pages > 1 && ty >= PAGER_Y &&
                       ty < PAGER_Y + 32) {
                if (tx < 120)
                    page = (page + pages - 1) % pages;
                else if (tx > LCD_W - 120)
                    page = (page + 1) % pages;
            }
        }

        /* society picker strip (y 96..122): tap a segment to switch view */
        if (!interview && live && tap && ty >= 96 && ty <= 122) {
            int seg = tx / (LCD_W / MAX_SOCIETIES);
            if (seg >= 0 && seg < st.n_societies)
                net_society_select(st.societies[seg].id);
        }

        /* session row: open the session list */
        int session_y = interview ? IV_SESSION_Y : 124;
        if (live && tap && ty >= session_y && ty <= session_y + 14)
            picking = true;
        else if (picking && st.conn == CONN_WS && tap && ty >= SP_Y) {
            int r = (ty - SP_Y) / SP_ROW;
            if (ty >= BTN_Y) {
                picking = false;
            } else if (r == 0 && st.n_runs > 0) {
                net_run_select(NULL);   /* follow the newest live session */
                picking = false;
            } else if (r >= 1 && r - 1 < st.n_runs && r < SP_ROWS) {
                net_run_select(st.runs[r - 1].id);
                picking = false;
            }
        }

        /* CONNECT (the only way a connection is ever opened) */
        if (tap && st.conn == CONN_WIFI && ty >= CONN_BOX_Y && ty < CONN_BOX_Y + CONN_BOX_H &&
            tx >= (LCD_W - CONN_BOX_W) / 2 && tx < (LCD_W + CONN_BOX_W) / 2) {
            net_ws_retry_now();
            retry_flash_until = now_us + 400000;
        }
        if (tap && st.conn == CONN_NOT_PAIRED) {
            STATE_LOCK();
            g_state.conn = CONN_WIFI;
            STATE_UNLOCK();
        }
        prev_touched = touched;

        gfx_begin_frame();

        if (st.conn == CONN_NOT_PAIRED) {
            draw_not_paired(&st);
        } else {
            if (st.voice == VOICE_DONE && !voice_done_at)
                voice_done_at = esp_timer_get_time();
            if (st.voice != VOICE_DONE)
                voice_done_at = 0;
            /* show transcript ~5 s, then back to idle */
            if (voice_done_at && esp_timer_get_time() - voice_done_at > 5000000) {
                STATE_LOCK();
                g_state.voice = VOICE_IDLE;
                g_state.transcript[0] = 0;
                STATE_UNLOCK();
                voice_done_at = 0;
            }

            draw_status(&st);
            draw_tabs(&st);
            if (st.conn != CONN_WS) {
                draw_connect(&st, esp_timer_get_time(), retry_flash_until);
            } else if (picking) {
                draw_session_picker(&st);
            } else {
                if (interview) {
                    draw_iv_header(&st);
                    draw_session(&st, IV_SESSION_Y + 2);
                    if (st.iv_agent.id < 0)
                        draw_picker(&st, page);
                    else
                        draw_convo(&st);
                } else {
                    draw_header(&st);
                    draw_society_strip(&st);
                    draw_session(&st, 126);
                    draw_threads(&st);
                    draw_canvas(&st);
                    draw_murmur(&st);
                }
                if (interview)
                    draw_iv_button(&st, held, held ? esp_timer_get_time() - held_since : 0);
                else
                    draw_button(&st, held, held ? esp_timer_get_time() - held_since : 0);
            }
            draw_toast(&st, esp_timer_get_time());
        }

        hw_present(gfx_fb());
        vTaskDelay(pdMS_TO_TICKS(66));   /* ~15 fps */
    }
}
