#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

void net_start(void);

bool net_ws_connected(void);

/* Send a text frame (JSON). Safe to call from any task; no-op if offline. */
int  net_ws_send_text(const char *json);

/* Send one binary frame of PCM (16-bit LE mono 16 kHz). */
int  net_ws_send_audio(const void *pcm, size_t len);

/* GET /devices/me -> fills run state; returns http status (0 = no connection). */
int  net_check_token(void);

/* Ask the server to show society N (0..2). Sent now and re-sent on reconnect. */
void net_society_select(int n);
int  net_society_selected(void);   /* user's pick, or -1 = token default */

/* Session (world/run) selection. id NULL/"" = follow newest live run. */
void net_run_select(const char *id);
const char *net_run_pinned(void);  /* pinned run id, or "" = following newest */

/* Ask ws_task to retry the connection now (wakes the manual-retry wait). */
void net_ws_retry_now(void);
/* 0 = waiting for tap, 1 = auto-retry backoff, 2 = handshake in progress */
void net_ws_phase(int *phase, int *attempt);

/* Screen mode (MODE_SOCIETY / MODE_INTERVIEW), persisted in NVS. */
void net_load_mode(void);
void net_set_mode(int mode);

/* Interview mode: ask for the run's agents, pick one to talk to. */
void net_agents_list(void);
void net_interview_select(int agent_id);
