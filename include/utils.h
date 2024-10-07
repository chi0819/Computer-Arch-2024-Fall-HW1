#ifndef UTILS_H
#define UTILS_H

#include "bf16.h"

float sigmoid(float x);

float *sample(int sample_size);

bf16_t bfmul16(bf16_t a, bf16_t b);

bf16_t bfdiv16(bf16_t a, bf16_t b);

bf16_t bfadd16(bf16_t a, bf16_t b);

#endif
