#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

/* Shared application state: written by net_task / mic task, read by ui_task.
 * Always hold g_state_mtx for multi-field access. */

typedef enum {
    CONN_OFFLINE,      /* no wifi */
    CONN_WIFI,         /* wifi up, server not connected */
    CONN_WS,           /* websocket connected */
    CONN_NOT_PAIRED,   /* server rejected token (401) */
} conn_state_t;

typedef enum {
    VOICE_IDLE,
    VOICE_LISTENING,     /* button held, streaming */
    VOICE_TRANSCRIBING,  /* server working */
    VOICE_DONE,          /* transcript ready */
    VOICE_ERROR,
} voice_state_t;

#define MAX_SOCIETIES 3
#define MAX_THREADS   3
#define MAX_PALETTE   16
#define MAX_RUNS      8
#define MAX_AGENTS    48
#define MAX_CONVO     6

typedef enum {
    MODE_SOCIETY,      /* influence a society with murmurs */
    MODE_INTERVIEW,    /* pick one agent and talk with it */
} screen_mode_t;

typedef enum {
    IV_IDLE,           /* nobody picked / waiting for input */
    IV_THINKING,       /* question sent, agent is answering */
    IV_SPEAKING,       /* reply audio playing */
} iv_status_t;

typedef struct {
    int  id;
    char name[28];
    char country[24];
    uint32_t color;
    int  society;      /* -1 = none */
} agent_info_t;

typedef struct {
    bool me;           /* true = the visitor, false = the agent */
    char text[400];
} convo_line_t;

typedef struct {
    char title[64];
    char author[32];
    int  replies;
} thread_t;

typedef struct {
    int  id;
    char name[32];
    int  members;
} society_brief_t;

typedef struct {
    char id[12];
    int  agents;
    char model[16];
    char soc_names[64];   /* comma-joined society names */
} run_info_t;

typedef struct {
    conn_state_t conn;
    screen_mode_t mode;

    /* interview mode */
    agent_info_t agents[MAX_AGENTS];
    int  n_agents;
    char agents_run[12];      /* run the agent list belongs to */
    agent_info_t iv_agent;    /* who we are talking to; id -1 = nobody */
    iv_status_t iv_status;
    convo_line_t convo[MAX_CONVO];
    int  n_convo;

    /* run + society */
    bool run_active;
    char run_id[12];          /* effective run id; "" when none */
    bool run_pinned;          /* user pinned a session (vs following newest) */
    run_info_t runs[MAX_RUNS];
    int  n_runs;
    char society_name[32];
    int  society_id;
    uint32_t society_color;   /* 0xRRGGBB */
    int  members;
    int  joined, left;        /* last 60 s migration counts */

    society_brief_t societies[MAX_SOCIETIES];
    int  n_societies;
    int  preview_society;     /* -1 = own; UI-only peek selection */

    thread_t threads[MAX_THREADS];
    int  n_threads;

    char murmur[192];
    bool has_murmur;

    voice_state_t voice;
    char transcript[192];
    int  mic_level;           /* 0..100, ui meter while listening */

    char palette[MAX_PALETTE][8];   /* "#rrggbb" */
    int  n_palette;

    uint8_t *canvas_idx;            /* w*h palette indices, PSRAM */
    int  canvas_w, canvas_h;
    bool canvas_fresh;

    char toast[96];
    int64_t toast_until_us;

    char mac[18];                   /* "AA:BB:CC:DD:EE:FF", for pairing screen */
} app_state_t;

extern app_state_t g_state;
extern SemaphoreHandle_t g_state_mtx;

#define STATE_LOCK()   xSemaphoreTake(g_state_mtx, portMAX_DELAY)
#define STATE_UNLOCK() xSemaphoreGive(g_state_mtx)
