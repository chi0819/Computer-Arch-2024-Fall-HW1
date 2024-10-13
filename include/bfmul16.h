#ifndef BFMUL16_H
#define BFMUL16_H

#include <stdint.h>
#include "bf16.h"

uint16_t imul16(int16_t a, int16_t b);

bf16_t bfmul16(bf16_t a, bf16_t b);

#endif
