#ifndef UTILS_H
#define UTILS_H

#include "bf16.h"

uint16_t getbit(uint16_t value, int n);

void printBit(void *num, size_t size);

bf16_t bfmul16(bf16_t a, bf16_t b);

uint16_t imul16(int16_t a, int16_t b);

bf16_t bfpow16(bf16_t x, int n);

float floatfact(int n);

bf16_t bfdiv16(bf16_t a, bf16_t b);

uint16_t udiv16(uint16_t dividendm, uint16_t divisor);

bf16_t bfadd16(bf16_t a, bf16_t b);

bf16_t bfsub16(bf16_t a, bf16_t b);

float sigmoid(float x);

float *sample(int sample_size);

#endif
