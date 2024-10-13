#include <stdint.h>
#include "bf16.h"
#include "bf16utils.h"

uint16_t imul16(int16_t a, int16_t b)
{
    uint16_t r = 0;

    for (int i = 0; i < 8; i++) {
        if (getbit(b, i))
            r += a << i;
    }
    return r;
}


bf16_t bfmul16(bf16_t a, bf16_t b)
{
    uint16_t ia = a.bits;
    uint16_t ib = b.bits;

    /* mantissa 7 bits */
    uint8_t ma = (ia & 0x7F) | 0x80;
    uint8_t mb = (ib & 0x7F) | 0x80;

    /* exponent 8 bits */
    uint8_t exp_a = (ia >> 7) & 0xFF;
    uint8_t exp_b = (ib >> 7) & 0xFF;

    /* sign */
    uint16_t sign = (ia ^ ib) & 0x8000;

    /* mantissa multiplication */
    /* extend uint8_t ma and mb to uint16_t */
    /* only need uint16_t to store the multiplication of ma and mb */
    /* the multiplication result won't exceed 16 bits */
    uint16_t m = imul16((int16_t)ma, (int16_t)mb);
    // uint16_t m = (uint16_t)ma * (uint16_t)mb;

    /* normalization */
    uint8_t mshift = (m >> 15) & 1; /* check if bit 15 is set */
    m >>= mshift; /* shift mantissa right by 1 if overflow */
    
    /* adjust exponent: (exp_a + exp_b - bias) + mshift */
    int16_t exp_r = (int16_t)exp_a + (int16_t)exp_b - 127 + (int16_t)mshift;

    /* only take care of underflow */
    /* aim to process small number */
    if (exp_r <= 0) exp_r = 0;

    /* combine sign, exponent, and mantissa */
    return (bf16_t) {.bits = sign | (((uint16_t)exp_r << 7) & 0x7F80) | ((m >> 7) & 0x7F)};
}
