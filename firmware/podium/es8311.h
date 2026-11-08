#pragma once

#include <Arduino.h>

bool es8311ReadChipId(uint8_t &chipId1, uint8_t &chipId2);
bool es8311Init();
