#include "net.h"
#include "state.h"
#include "secrets.h"
#include "mic.h"

#include <string.h>
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_log.h"
#include "esp_event.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_http_client.h"
#include "esp_websocket_client.h"
#include "esp_crt_bundle.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "cJSON.h"

static const char *TAG = "net";

static EventGroupHandle_t s_eg;
#define EG_WIFI_UP BIT0

static esp_websocket_client_handle_t s_ws;
static volatile bool s_ws_connected;
static int s_selected = -1;   /* society picked by user; -1 = token default */
static TaskHandle_t s_ws_task;
static const int s_backoff_s[] = {1, 2, 5, 10};
static bool s_run_resynced;        /* pinned-run re-select done this connection */
static volatile bool s_retry_req;  /* human tapped CONNECT */
static volatile int  s_ws_phase;   /* 0=awaiting tap, 2=handshake now, 3=last attempt failed */
static volatile int  s_ws_attempt;
static volatile bool s_canvas_req;  /* human tapped the canvas */
static volatile bool s_ws_failed;  /* handshake or connection ended (error/disconnect event) */
static uint8_t s_last_op;          /* opcode of the frame being received */
static char *s_txt;                /* reassembly buffer for text frames split across events */
static int s_txt_len;
#define TXT_MAX (24 * 1024)

static void ws_task(void *arg);
static void canvas_task(void *arg);

/* ---------------- helpers ---------------- */

static void jstrncpy(char *dst, size_t n, const cJSON *obj, const char *key)
{
    const cJSON *s = cJSON_GetObjectItem(obj, key);
    if (cJSON_IsString(s) && s->valuestring) {
        strncpy(dst, s->valuestring, n - 1);
        dst[n - 1] = 0;
    }
}

static uint32_t hex_color(const char *s)
{
    if (!s || s[0] != '#')
        return 0xFFFFFF;
    return (uint32_t)strtoul(s + 1, NULL, 16) & 0xFFFFFF;
}

/* ---------------- nvs (persisted pinned run) ---------------- */

static void nvs_save_run(const char *id)
{
    nvs_handle_t h;
    if (nvs_open("w3p", NVS_READWRITE, &h) != ESP_OK)
        return;
    if (id && id[0])
        nvs_set_str(h, "run", id);
    else
        nvs_erase_key(h, "run");
    nvs_commit(h);
    nvs_close(h);
}

static void nvs_load_run(char *out, size_t n)
{
    out[0] = 0;
    nvs_handle_t h;
    if (nvs_open("w3p", NVS_READONLY, &h) != ESP_OK)
        return;
    size_t len = n;
    nvs_get_str(h, "run", out, &len);
    nvs_close(h);
}

void net_load_mode(void)
{
    nvs_handle_t h;
    uint8_t mode = MODE_SOCIETY;
    if (nvs_open("w3p", NVS_READONLY, &h) == ESP_OK) {
        nvs_get_u8(h, "mode", &mode);
        nvs_close(h);
    }
    g_state.mode = mode == MODE_INTERVIEW ? MODE_INTERVIEW : MODE_SOCIETY;
}

void net_set_mode(int mode)
{
    STATE_LOCK();
    g_state.mode = mode == MODE_INTERVIEW ? MODE_INTERVIEW : MODE_SOCIETY;
    STATE_UNLOCK();
    nvs_handle_t h;
    if (nvs_open("w3p", NVS_READWRITE, &h) == ESP_OK) {
        nvs_set_u8(h, "mode", (uint8_t)mode);
        nvs_commit(h);
        nvs_close(h);
    }
    ESP_LOGI(TAG, "mode -> %s", mode == MODE_INTERVIEW ? "interview" : "society");
}

static bool run_id_known(const char *id)
{
    STATE_LOCK();
    bool found = false;
    for (int i = 0; i < g_state.n_runs; i++)
        if (!strcmp(g_state.runs[i].id, id)) { found = true; break; }
    STATE_UNLOCK();
    return found;
}

/* ---------------- message parsing ---------------- */

static void parse_runs(const cJSON *arr)
{
    STATE_LOCK();
    g_state.n_runs = 0;
    if (cJSON_IsArray(arr)) {
        const cJSON *r;
        cJSON_ArrayForEach(r, arr) {
            if (g_state.n_runs >= MAX_RUNS)
                break;
            run_info_t *ri = &g_state.runs[g_state.n_runs];
            jstrncpy(ri->id, sizeof(ri->id), r, "id");
            ri->agents = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(r, "agents"));
            jstrncpy(ri->model, sizeof(ri->model), r, "model");
            /* join society names */
            ri->soc_names[0] = 0;
            const cJSON *sn = cJSON_GetObjectItem(r, "societies");
            if (cJSON_IsArray(sn)) {
                const cJSON *s;
                cJSON_ArrayForEach(s, sn) {
                    if (!cJSON_IsString(s)) continue;
                    size_t off = strlen(ri->soc_names);
                    size_t left = sizeof(ri->soc_names) - off;
                    snprintf(ri->soc_names + off, left, "%s%s",
                             off ? "," : "", s->valuestring);
                }
            }
            g_state.n_runs++;
        }
    }
    STATE_UNLOCK();
}

static void handle_hello(const cJSON *root)
{
    const cJSON *pal = cJSON_GetObjectItem(root, "palette");
    STATE_LOCK();
    g_state.n_palette = 0;
    if (cJSON_IsArray(pal)) {
        const cJSON *c;
        cJSON_ArrayForEach(c, pal) {
            if (g_state.n_palette >= MAX_PALETTE)
                break;
            if (cJSON_IsString(c))
                strncpy(g_state.palette[g_state.n_palette++], c->valuestring, 7);
        }
    }
    g_state.society_id = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(root, "society"));
    g_state.run_id[0] = 0;
    jstrncpy(g_state.run_id, sizeof(g_state.run_id), root, "run");
    g_state.run_active = g_state.run_id[0] != 0;
    STATE_UNLOCK();
    parse_runs(cJSON_GetObjectItem(root, "runs"));
    ESP_LOGI(TAG, "hello: society=%d palette=%d run=%s runs=%d",
             g_state.society_id, g_state.n_palette, g_state.run_id, g_state.n_runs);

    /* once per connection: re-pin the stored session if it still exists */
    if (!s_run_resynced) {
        s_run_resynced = true;
        char stored[16];
        nvs_load_run(stored, sizeof(stored));
        if (stored[0]) {
            if (run_id_known(stored))
                net_run_select(stored);
            else {
                nvs_save_run("");
                ESP_LOGW(TAG, "stored run %s gone, following newest", stored);
            }
        }
    }
}

static void handle_state(const cJSON *root)
{
    STATE_LOCK();
    const cJSON *runj = cJSON_GetObjectItem(root, "run");
    if (cJSON_IsString(runj) && runj->valuestring) {
        strncpy(g_state.run_id, runj->valuestring, sizeof(g_state.run_id) - 1);
        g_state.run_id[sizeof(g_state.run_id) - 1] = 0;
        g_state.run_active = true;
    } else {
        g_state.run_id[0] = 0;
        g_state.run_active = false;
    }

    const cJSON *soc = cJSON_GetObjectItem(root, "society");
    if (cJSON_IsObject(soc)) {
        g_state.society_id = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(soc, "id"));
        jstrncpy(g_state.society_name, sizeof(g_state.society_name), soc, "name");
        g_state.society_color = hex_color(cJSON_GetStringValue(cJSON_GetObjectItem(soc, "color")));
        g_state.members = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(soc, "members"));
        g_state.joined  = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(soc, "joined"));
        g_state.left    = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(soc, "left"));
    }

    const cJSON *arr = cJSON_GetObjectItem(root, "societies");
    g_state.n_societies = 0;
    if (cJSON_IsArray(arr)) {
        const cJSON *s;
        cJSON_ArrayForEach(s, arr) {
            if (g_state.n_societies >= MAX_SOCIETIES)
                break;
            society_brief_t *b = &g_state.societies[g_state.n_societies];
            b->id = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(s, "id"));
            jstrncpy(b->name, sizeof(b->name), s, "name");
            b->members = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(s, "members"));
            g_state.n_societies++;
        }
    }

    arr = cJSON_GetObjectItem(root, "threads");
    g_state.n_threads = 0;
    if (cJSON_IsArray(arr)) {
        const cJSON *t;
        cJSON_ArrayForEach(t, arr) {
            if (g_state.n_threads >= MAX_THREADS)
                break;
            thread_t *th = &g_state.threads[g_state.n_threads];
            jstrncpy(th->title, sizeof(th->title), t, "title");
            jstrncpy(th->author, sizeof(th->author), t, "author");
            th->replies = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(t, "replies"));
            g_state.n_threads++;
        }
    }

    const cJSON *m = cJSON_GetObjectItem(root, "murmur");
    g_state.has_murmur = cJSON_IsString(m) && m->valuestring && m->valuestring[0];
    if (g_state.has_murmur) {
        strncpy(g_state.murmur, m->valuestring, sizeof(g_state.murmur) - 1);
        g_state.murmur[sizeof(g_state.murmur) - 1] = 0;
    }

    const cJSON *v = cJSON_GetObjectItem(root, "voice");
    if (cJSON_IsObject(v)) {
        const char *st = cJSON_GetStringValue(cJSON_GetObjectItem(v, "status"));
        if (st) {
            if (!strcmp(st, "transcribing"))      g_state.voice = VOICE_TRANSCRIBING;
            else if (!strcmp(st, "done"))       g_state.voice = VOICE_DONE;
            else if (!strcmp(st, "error")) {
                g_state.voice = VOICE_ERROR;
                const char *em = cJSON_GetStringValue(cJSON_GetObjectItem(v, "error"));
                ESP_LOGW(TAG, "server voice error: %s", em ? em : "(none)");
            }
            else if (!strcmp(st, "idle") && g_state.voice != VOICE_LISTENING)
                g_state.voice = VOICE_IDLE;
        }
        const cJSON *tr = cJSON_GetObjectItem(v, "transcript");
        if (cJSON_IsString(tr) && tr->valuestring && tr->valuestring[0]) {
            strncpy(g_state.transcript, tr->valuestring, sizeof(g_state.transcript) - 1);
            g_state.transcript[sizeof(g_state.transcript) - 1] = 0;
        }
    }
    STATE_UNLOCK();
}

static void handle_migration(const cJSON *root)
{
    const char *agent = cJSON_GetStringValue(cJSON_GetObjectItem(root, "agent"));
    int from = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(root, "from"));
    int to = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(root, "to"));
    STATE_LOCK();
    if (to == g_state.society_id)
        snprintf(g_state.toast, sizeof(g_state.toast), "%s joined from Society %d",
                 agent ? agent : "Someone", from + 1);
    else if (from == g_state.society_id)
        snprintf(g_state.toast, sizeof(g_state.toast), "%s left for Society %d",
                 agent ? agent : "Someone", to + 1);
    else
        snprintf(g_state.toast, sizeof(g_state.toast), "%s moved %d -> %d",
                 agent ? agent : "Someone", from + 1, to + 1);
    g_state.toast_until_us = esp_timer_get_time() + 4000000;
    STATE_UNLOCK();
}

/* ---------------- interview mode ---------------- */

static void parse_agent(const cJSON *a, agent_info_t *out)
{
    memset(out, 0, sizeof(*out));
    out->id = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(a, "id"));
    jstrncpy(out->name, sizeof(out->name), a, "name");
    jstrncpy(out->country, sizeof(out->country), a, "country");
    out->color = hex_color(cJSON_GetStringValue(cJSON_GetObjectItem(a, "color")));
    const cJSON *s = cJSON_GetObjectItem(a, "society");
    out->society = cJSON_IsNumber(s) ? (int)s->valuedouble : -1;
}

/* caller holds the state lock */
static void convo_push(bool me, const char *text)
{
    if (g_state.n_convo == MAX_CONVO) {
        memmove(&g_state.convo[0], &g_state.convo[1], sizeof(convo_line_t) * (MAX_CONVO - 1));
        g_state.n_convo--;
    }
    convo_line_t *l = &g_state.convo[g_state.n_convo++];
    l->me = me;
    snprintf(l->text, sizeof(l->text), "%s", text ? text : "");
}

/* caller holds the state lock */
static void toast(const char *msg)
{
    snprintf(g_state.toast, sizeof(g_state.toast), "%s", msg);
    g_state.toast_until_us = esp_timer_get_time() + 4000000;
}

static void handle_agents(const cJSON *root)
{
    const cJSON *arr = cJSON_GetObjectItem(root, "agents");
    const char *run = cJSON_GetStringValue(cJSON_GetObjectItem(root, "run"));
    STATE_LOCK();
    if (g_state.agents_run[0] && (!run || strcmp(run, g_state.agents_run))) {
        g_state.iv_agent.id = -1;   /* new world: the old interviewee is gone */
        g_state.n_convo = 0;
        g_state.iv_status = IV_IDLE;
    }
    snprintf(g_state.agents_run, sizeof(g_state.agents_run), "%s", run ? run : "");
    g_state.n_agents = 0;
    if (cJSON_IsArray(arr)) {
        const cJSON *a;
        cJSON_ArrayForEach(a, arr) {
            if (g_state.n_agents >= MAX_AGENTS)
                break;
            parse_agent(a, &g_state.agents[g_state.n_agents++]);
        }
    }
    STATE_UNLOCK();
    ESP_LOGI(TAG, "agents: %d in run %s", g_state.n_agents, run ? run : "-");
}

static void handle_interview(const char *type, const cJSON *root)
{
    const char *text = cJSON_GetStringValue(cJSON_GetObjectItem(root, "text"));
    if (!strcmp(type, "interview.agent")) {
        STATE_LOCK();
        parse_agent(cJSON_GetObjectItem(root, "agent"), &g_state.iv_agent);
        g_state.n_convo = 0;
        g_state.iv_status = IV_IDLE;
        char msg[64];
        snprintf(msg, sizeof(msg), "talking to %s", g_state.iv_agent.name);
        toast(msg);
        STATE_UNLOCK();
        ESP_LOGI(TAG, "interviewing agent %d", g_state.iv_agent.id);
    } else if (!strcmp(type, "interview.heard")) {
        STATE_LOCK();
        convo_push(true, text);
        g_state.iv_status = IV_THINKING;
        STATE_UNLOCK();
    } else if (!strcmp(type, "interview.reply")) {
        bool audio = cJSON_IsTrue(cJSON_GetObjectItem(root, "audio"));
        if (audio)
            spk_begin((int)cJSON_GetNumberValue(cJSON_GetObjectItem(root, "bytes")));
        STATE_LOCK();
        convo_push(false, text);
        g_state.iv_status = audio ? IV_SPEAKING : IV_IDLE;
        STATE_UNLOCK();
    } else if (!strcmp(type, "interview.audio_end")) {
        spk_end();
    } else if (!strcmp(type, "interview.error")) {
        const char *em = cJSON_GetStringValue(cJSON_GetObjectItem(root, "error"));
        ESP_LOGW(TAG, "interview error: %s", em ? em : "?");
        STATE_LOCK();
        toast(em ? em : "something went wrong");
        if (g_state.iv_status == IV_THINKING)
            g_state.iv_status = IV_IDLE;
        STATE_UNLOCK();
    }
}

void net_agents_list(void)
{
    net_ws_send_text("{\"type\":\"agents.list\"}");
}

void net_interview_select(int agent_id)
{
    char msg[64];
    snprintf(msg, sizeof(msg), "{\"type\":\"interview.select\",\"agent\":%d}", agent_id);
    net_ws_send_text(msg);
}

static void on_ws_text(const char *data, int len)
{
    cJSON *root = cJSON_ParseWithLength(data, len);
    if (!root) {
        ESP_LOGW(TAG, "bad json: %.*s", len > 80 ? 80 : len, data);
        return;
    }
    const char *type = cJSON_GetStringValue(cJSON_GetObjectItem(root, "type"));
    if (type) {
        if (!strcmp(type, "hello"))         handle_hello(root);
        else if (!strcmp(type, "state"))    handle_state(root);
        else if (!strcmp(type, "migration")) handle_migration(root);
        else if (!strcmp(type, "society.selected"))
            ESP_LOGI(TAG, "viewing society %d",
                     (int)cJSON_GetNumberValue(cJSON_GetObjectItem(root, "society")));
        else if (!strcmp(type, "runs"))          parse_runs(cJSON_GetObjectItem(root, "runs"));
        else if (!strcmp(type, "agents"))        handle_agents(root);
        else if (!strncmp(type, "interview.", 10)) handle_interview(type, root);
        else if (!strcmp(type, "run.selected")) {
            bool pinned = cJSON_IsTrue(cJSON_GetObjectItem(root, "pinned"));
            STATE_LOCK();
            jstrncpy(g_state.run_id, sizeof(g_state.run_id), root, "run");
            g_state.run_active = g_state.run_id[0] != 0;
            g_state.run_pinned = pinned;
            STATE_UNLOCK();
            if (pinned && g_state.run_id[0])
                nvs_save_run(g_state.run_id);
            else
                nvs_save_run("");
            ESP_LOGI(TAG, "run.selected -> %s (pinned=%d)", g_state.run_id, pinned);
        }
        else if (!strcmp(type, "error")) {
            const char *em = cJSON_GetStringValue(cJSON_GetObjectItem(root, "error"));
            ESP_LOGW(TAG, "server error: %s", em ? em : "?");
        }
    }
    cJSON_Delete(root);
}

/* ---------------- websocket ---------------- */

static void ws_event_handler(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    esp_websocket_event_data_t *d = (esp_websocket_event_data_t *)data;
    switch (id) {
    case WEBSOCKET_EVENT_CONNECTED:
        ESP_LOGI(TAG, "ws connected");
        s_ws_connected = true;
        STATE_LOCK();
        g_state.conn = CONN_WS;
        STATE_UNLOCK();
        /* server doesn't remember society choice across connections */
        if (s_selected >= 0)
            net_society_select(s_selected);
        break;
    case WEBSOCKET_EVENT_DISCONNECTED:
        ESP_LOGW(TAG, "ws disconnected");
        s_ws_connected = false;
        s_ws_failed = true;
        s_run_resynced = false;
        break;
    case WEBSOCKET_EVENT_DATA:
        if (d->op_code == 0x1 || d->op_code == 0x2)
            s_last_op = d->op_code;
        else if (d->op_code != 0x0)
            break;   /* ping / pong / close */
        if (!d->data_ptr || d->data_len <= 0)
            break;
        if (s_last_op == 0x2) {
            spk_feed(d->data_ptr, d->data_len);   /* interview reply audio */
        } else if (s_last_op == 0x1) {
            if (d->payload_offset == 0 && d->data_len >= d->payload_len) {
                on_ws_text(d->data_ptr, d->data_len);
                break;
            }
            /* a text frame larger than the client buffer arrives in pieces */
            if (!s_txt)
                s_txt = heap_caps_malloc(TXT_MAX, MALLOC_CAP_SPIRAM);
            if (!s_txt)
                break;
            if (d->payload_offset == 0)
                s_txt_len = 0;
            if (s_txt_len + d->data_len <= TXT_MAX) {
                memcpy(s_txt + s_txt_len, d->data_ptr, d->data_len);
                s_txt_len += d->data_len;
            }
            if (d->payload_offset + d->data_len >= d->payload_len) {
                on_ws_text(s_txt, s_txt_len);
                s_txt_len = 0;
            }
        }
        break;
    case WEBSOCKET_EVENT_ERROR:
        ESP_LOGW(TAG, "ws error type=%d status=%d",
                 d->error_handle.error_type, d->error_handle.esp_ws_handshake_status_code);
        s_ws_connected = false;
        s_ws_failed = true;
        break;
    default:
        break;
    }
}

static void ws_task(void *arg)
{
    char uri[256];
    const char *scheme = "ws";
    const char *p = SERVER_URL;
    if (!strncmp(p, "https://", 8)) { scheme = "wss"; p += 8; }
    else if (!strncmp(p, "http://", 7)) { p += 7; }
    snprintf(uri, sizeof(uri), "%s://%s/devices/ws", scheme, p);   /* token goes in the header, not the logged URL */
    char ws_auth[160];
    snprintf(ws_auth, sizeof(ws_auth), "Authorization: Bearer %s\r\n", DEVICE_TOKEN);  /* raw header lines need CRLF */

    while (1) {
        if (!(xEventGroupGetBits(s_eg) & EG_WIFI_UP)) {
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }
        /* only a human tap on CONNECT opens a connection: no automatic retries */
        if (!s_retry_req) {
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }
        s_retry_req = false;
        STATE_LOCK();
        g_state.conn = CONN_WIFI;
        STATE_UNLOCK();

        esp_websocket_client_config_t cfg = {
            .uri = uri,
            .buffer_size = 8192,
            .disable_auto_reconnect = true,
            .network_timeout_ms = 30000,   /* cold starts can be slow */
            .ping_interval_sec = 15,
            .headers = ws_auth,
            .crt_bundle_attach = esp_crt_bundle_attach,
        };
        s_ws_attempt++;
        s_ws_phase = 2;
        s_ws = esp_websocket_client_init(&cfg);
        esp_websocket_register_events(s_ws, WEBSOCKET_EVENT_ANY, ws_event_handler, NULL);
        ESP_LOGI(TAG, "ws connecting %s://%s/devices/ws", scheme, p);
        s_ws_failed = false;
        esp_err_t err = esp_websocket_client_start(s_ws);
        if (err != ESP_OK)
            ESP_LOGW(TAG, "ws start failed: %s", esp_err_to_name(err));

        /* start() is asynchronous: wait for the handshake to succeed or fail */
        for (int i = 0; err == ESP_OK && i < 35000 / 100 && !s_ws_connected && !s_ws_failed; i++) {
            if (!(xEventGroupGetBits(s_eg) & EG_WIFI_UP))
                break;
            vTaskDelay(pdMS_TO_TICKS(100));
        }
        bool ok = s_ws_connected;

        /* stay connected until the server or wifi drops us */
        while (s_ws_connected || esp_websocket_client_is_connected(s_ws)) {
            if (!(xEventGroupGetBits(s_eg) & EG_WIFI_UP))
                break;
            vTaskDelay(pdMS_TO_TICKS(200));
        }
        s_ws_connected = false;
        esp_websocket_client_stop(s_ws);
        esp_websocket_client_destroy(s_ws);
        s_ws = NULL;

        STATE_LOCK();
        if (g_state.conn == CONN_WS)
            g_state.conn = CONN_WIFI;
        STATE_UNLOCK();
        s_ws_phase = ok ? 0 : 3;
        if (ok)
            ESP_LOGW(TAG, "ws closed - tap CONNECT");
        else
            ESP_LOGW(TAG, "ws connect failed - tap CONNECT");

        /* one token check per failed attempt, to tell "not paired" from "server down" */
        if (!ok && (xEventGroupGetBits(s_eg) & EG_WIFI_UP) && net_check_token() == 401) {
            STATE_LOCK();
            g_state.conn = CONN_NOT_PAIRED;
            STATE_UNLOCK();
            ESP_LOGE(TAG, "server rejected token (401) - pair this device");
        }
    }
}

void net_canvas_refresh(void)
{
    s_canvas_req = true;
}

bool net_ws_connected(void)
{
    return s_ws_connected;
}

int net_ws_send_text(const char *json)
{
    if (!s_ws_connected || !s_ws)
        return -1;
    return esp_websocket_client_send_text(s_ws, json, strlen(json), pdMS_TO_TICKS(500));
}

int net_ws_send_audio(const void *pcm, size_t len)
{
    if (!s_ws_connected || !s_ws)
        return -1;
    return esp_websocket_client_send_bin(s_ws, pcm, len, pdMS_TO_TICKS(500));
}

void net_society_select(int n)
{
    s_selected = n;
    if (!s_ws_connected)
        return;
    char msg[48];
    snprintf(msg, sizeof(msg), "{\"type\":\"society.select\",\"society\":%d}", n);
    net_ws_send_text(msg);
}

int net_society_selected(void)
{
    return s_selected;
}

void net_run_select(const char *id)
{
    char msg[64];
    if (id && id[0])
        snprintf(msg, sizeof(msg), "{\"type\":\"run.select\",\"run\":\"%s\"}", id);
    else
        snprintf(msg, sizeof(msg), "{\"type\":\"run.select\",\"run\":null}");
    net_ws_send_text(msg);
}

const char *net_run_pinned(void)
{
    return g_state.run_pinned ? g_state.run_id : "";
}

void net_ws_retry_now(void)
{
    s_retry_req = true;
}

void net_ws_phase(int *phase, int *attempt)
{
    *phase = s_ws_phase;
    *attempt = s_ws_attempt;
}

/* ---------------- http ---------------- */

static int http_get(const char *path, uint8_t *buf, int cap, int *out_len)
{
    char url[256];
    snprintf(url, sizeof(url), "%s%s", SERVER_URL, path);
    esp_http_client_config_t cfg = {
        .url = url,
        .timeout_ms = 5000,
        .crt_bundle_attach = esp_crt_bundle_attach,
    };
    esp_http_client_handle_t c = esp_http_client_init(&cfg);
    char auth[160];
    snprintf(auth, sizeof(auth), "Bearer %s", DEVICE_TOKEN);
    esp_http_client_set_header(c, "Authorization", auth);
    esp_err_t err = esp_http_client_open(c, 0);
    int status = 0;
    if (err == ESP_OK) {
        esp_http_client_fetch_headers(c);   /* parses the status line */
        status = esp_http_client_get_status_code(c);
        int len = 0;
        if (buf) {
            int r;
            while (len < cap && (r = esp_http_client_read(c, (char *)buf + len, cap - len)) > 0)
                len += r;
        }
        if (out_len)
            *out_len = len;
    } else {
        ESP_LOGW(TAG, "GET %s failed: %s", path, esp_err_to_name(err));
    }
    esp_http_client_cleanup(c);
    return status;
}

int net_check_token(void)
{
    char body[256];
    int len = 0;
    char path[64];
    const char *rp = net_run_pinned();
    if (rp[0])
        snprintf(path, sizeof(path), "/devices/me?run=%s", rp);
    else
        snprintf(path, sizeof(path), "/devices/me");
    int status = http_get(path, (uint8_t *)body, sizeof(body) - 1, &len);
    if (status == 200 && len > 0) {
        body[len] = 0;
        cJSON *root = cJSON_Parse(body);
        if (root) {
            STATE_LOCK();
            g_state.run_active = !cJSON_IsNull(cJSON_GetObjectItem(root, "run"));
            g_state.society_id = (int)cJSON_GetNumberValue(cJSON_GetObjectItem(root, "society"));
            STATE_UNLOCK();
            cJSON_Delete(root);
        }
    }
    return status;
}

/* canvas thumbnail poller (optional but cheap) */
static void canvas_task(void *arg)
{
    uint8_t *buf = heap_caps_malloc(240 * 160, MALLOC_CAP_SPIRAM);
    if (!buf)
        buf = heap_caps_malloc(240 * 160, MALLOC_CAP_8BIT);
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(200));
        if (!s_canvas_req)
            continue;   /* only a tap on the canvas fetches it */
        s_canvas_req = false;
        if (!s_ws_connected || !buf)
            continue;
        int len = 0;
        char path[96];
        const char *rp = (g_state.run_pinned && g_state.run_id[0]) ? g_state.run_id : NULL;
        if (s_selected >= 0 && rp)
            snprintf(path, sizeof(path), "/devices/canvas?w=240&h=160&society=%d&run=%s", s_selected, rp);
        else if (s_selected >= 0)
            snprintf(path, sizeof(path), "/devices/canvas?w=240&h=160&society=%d", s_selected);
        else if (rp)
            snprintf(path, sizeof(path), "/devices/canvas?w=240&h=160&run=%s", rp);
        else
            snprintf(path, sizeof(path), "/devices/canvas?w=240&h=160");
        int status = http_get(path, buf, 240 * 160, &len);
        if (status == 200 && len > 0) {
            STATE_LOCK();
            if (!g_state.canvas_idx)
                g_state.canvas_idx = heap_caps_malloc(240 * 160, MALLOC_CAP_SPIRAM);
            if (g_state.canvas_idx) {
                int w = 240, h = len / 240;
                if (h > 160) h = 160;
                memcpy(g_state.canvas_idx, buf, w * h);
                g_state.canvas_w = w;
                g_state.canvas_h = h;
                g_state.canvas_fresh = true;
            }
            STATE_UNLOCK();
        }
    }
}

/* ---------------- wifi ---------------- */

static esp_timer_handle_t s_wifi_retry;
static int s_wifi_fails;

static void wifi_retry_cb(void *arg)
{
    esp_wifi_connect();
}

static void scan_dump_task(void *arg)
{
    wifi_scan_config_t sc = {.show_hidden = true};
    if (esp_wifi_scan_start(&sc, true) == ESP_OK) {
        uint16_t n = 0;
        esp_wifi_scan_get_ap_num(&n);
        ESP_LOGW(TAG, "scan found %d APs:", n);
        wifi_ap_record_t aps[20];
        uint16_t got = 20;
        if (esp_wifi_scan_get_ap_records(&got, aps) == ESP_OK) {
            for (int i = 0; i < got; i++)
                ESP_LOGW(TAG, "  \"%s\" rssi=%d auth=%d",
                         aps[i].ssid, aps[i].rssi, aps[i].authmode);
        }
    }
    vTaskDelete(NULL);
}

static void wifi_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_event_sta_disconnected_t *d = (wifi_event_sta_disconnected_t *)data;
        ESP_LOGW(TAG, "wifi disconnected, reason=%d", d->reason);
        xEventGroupClearBits(s_eg, EG_WIFI_UP);
        STATE_LOCK();
        g_state.conn = CONN_OFFLINE;
        STATE_UNLOCK();
        s_wifi_fails++;
        if (s_wifi_fails == 3 || s_wifi_fails == 8) {
            /* diagnostic: what can we actually see at 2.4GHz?
             * must run in its own task - a blocking scan inside the event
             * handler deadlocks the event task */
            xTaskCreate(scan_dump_task, "scandump", 4096, NULL, 3, NULL);
        }
        int delay_s = s_backoff_s[s_wifi_fails < 4 ? s_wifi_fails - 1 : 3];
        if (s_wifi_fails > 4)
            delay_s = 10;
        ESP_LOGW(TAG, "wifi down, retry in %ds", delay_s);
        esp_timer_start_once(s_wifi_retry, (int64_t)delay_s * 1000000);
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "got ip " IPSTR, IP2STR(&e->ip_info.ip));
        s_wifi_fails = 0;
        xEventGroupSetBits(s_eg, EG_WIFI_UP);
        STATE_LOCK();
        g_state.conn = CONN_WIFI;
        STATE_UNLOCK();
    }
}

void net_start(void)
{
    s_eg = xEventGroupCreate();

    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t wicfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wicfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event, NULL));

    const esp_timer_create_args_t tcfg = {.callback = wifi_retry_cb, .name = "wifi_retry"};
    esp_timer_create(&tcfg, &s_wifi_retry);

    wifi_config_t cfg = {0};
    strncpy((char *)cfg.sta.ssid, WIFI_SSID, sizeof(cfg.sta.ssid) - 1);
    strncpy((char *)cfg.sta.password, WIFI_PASSWORD, sizeof(cfg.sta.password) - 1);
    cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    cfg.sta.pmf_cfg.capable = true;          /* WPA3-transition APs may ask for PMF */
    cfg.sta.pmf_cfg.required = false;
    cfg.sta.sae_pwe_h2e = WPA3_SAE_PWE_BOTH;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &cfg));
    ESP_ERROR_CHECK(esp_wifi_start());

    xTaskCreate(ws_task, "ws", 6144, NULL, 5, &s_ws_task);
    xTaskCreate(canvas_task, "canvas", 6144, NULL, 4, NULL);
}
