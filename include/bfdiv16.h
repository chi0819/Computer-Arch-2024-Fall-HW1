#ifndef BFDIV16_H
#define BFDIV16_H

#include <stdint.h>
#include "bf16.h"

uint16_t idiv7(int16_t a, int16_t b);

bf16_t bfdiv16(bf16_t a, bf16_t b);

#endif
