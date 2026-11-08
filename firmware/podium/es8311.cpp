#include "es8311.h"

#include <Wire.h>

namespace {

constexpr uint8_t ES8311_ADDRESS = 0x18;

struct RegisterValue {
  uint8_t reg;
  uint8_t value;
};

bool writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(ES8311_ADDRESS);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool readRegister(uint8_t reg, uint8_t &value) {
  Wire.beginTransmission(ES8311_ADDRESS);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(ES8311_ADDRESS, static_cast<uint8_t>(1)) != 1) return false;
  value = Wire.read();
  return true;
}

}

bool es8311ReadChipId(uint8_t &chipId1, uint8_t &chipId2) {
  return readRegister(0xFD, chipId1) && readRegister(0xFE, chipId2);
}

bool es8311Init() {
  // Register sequence follows Espressif esp-adf/components/esp_codec_dev/device/es8311/es8311.c (es8311_open/es8311_start).
  if (!writeRegister(0x00, 0x1F)) return false;
  delay(10);
  if (!writeRegister(0x00, 0x00)) return false;
  delay(10);

  static constexpr RegisterValue registers[] = {
      {0x0D, 0xFA}, {0x44, 0x08}, {0x44, 0x08},
      {0x01, 0x30}, {0x02, 0x00}, {0x03, 0x10}, {0x16, 0x24},
      {0x04, 0x10}, {0x05, 0x00}, {0x0B, 0x00}, {0x0C, 0x00},
      {0x10, 0x1F}, {0x11, 0x7F}, {0x00, 0x80}, {0x00, 0x80},
      {0x01, 0x3F}, {0x02, 0x00}, {0x05, 0x00}, {0x03, 0x10},
      {0x04, 0x10}, {0x07, 0x00}, {0x08, 0xFF}, {0x06, 0x03},
      {0x09, 0x0C}, {0x0A, 0x0C}, {0x13, 0x10}, {0x1B, 0x0A},
      {0x1C, 0x6A}, {0x44, 0x58},
      {0x09, 0x0C}, {0x0A, 0x0C}, {0x17, 0xBF}, {0x0E, 0x02},
      {0x12, 0x00}, {0x14, 0x1A}, {0x0D, 0x01}, {0x15, 0x40},
      {0x37, 0x08}, {0x45, 0x00}, {0x32, 0xBF},
  };

  for (const RegisterValue &entry : registers) {
    if (!writeRegister(entry.reg, entry.value)) return false;
  }

  uint8_t mute = 0;
  if (!readRegister(0x31, mute)) return false;
  return writeRegister(0x31, mute & 0x9F);
}
