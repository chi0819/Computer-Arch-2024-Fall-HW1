#ifndef CONVERSION_H
#define CONVERSION_H

#include "bf16.h"

bf16_t fp32_to_bf16(float s);

float bf16_to_fp32(bf16_t h);

#endif
