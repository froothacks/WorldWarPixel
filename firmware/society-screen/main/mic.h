#pragma once

#include <stdbool.h>
#include <stddef.h>

/* Bring up the ES8311 codec (shared I2C bus) + I2S RX at 16 kHz mono.
 * Returns false (and disables voice features) if the codec/mic path fails. */
bool mic_init(void);

/* Called by the audio task; do not call directly. */
void mic_ptt_task(void *arg);

/* PTT control, safe from the UI task. */
void mic_ptt_press(void);
void mic_ptt_release(void);
bool mic_ptt_held(void);

/* Speaker (interview replies): 16 kHz mono s16le, same I2S port as the mic.
 * spk_begin starts a new clip (dropping any old one), spk_feed appends network data,
 * spk_end marks the clip complete, spk_stop cuts it off. */
void spk_begin(int bytes);
void spk_feed(const void *pcm, size_t len);
void spk_end(void);
void spk_stop(void);
void spk_task(void *arg);
