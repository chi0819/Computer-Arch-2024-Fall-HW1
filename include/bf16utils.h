#ifndef BF16UTILS_H
#define BF16UTILS_H

#include <stdlib.h>
#include <stdint.h>
#include "bf16.h"

uint16_t getbit(uint16_t value, int n);

void printBit(void *num, size_t size);

bf16_t bfpow16(bf16_t x, int n);

float floatfact(int n);

int my_clz(uint16_t x);

/* STL sigmoid approach */
float sigmoid(float x);

/* random sample float */
float *sample(int sample_size, float min, float max);

#endif
