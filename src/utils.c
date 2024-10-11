#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "bf16.h"

uint16_t getbit(uint16_t value, int n)
{
    return (value >> n) & 1;
}


void printBit(void *num, size_t size) {
    unsigned char *byte = (unsigned char *)num;  // Treat the input as bytes
    for (size_t i = 0; i < size; i++) {
        // Print each byte from most significant to least significant
        for (int j = 7; j >= 0; j--) {
            // Shift the byte j positions to the right and mask the least significant bit
            unsigned char bit = (byte[size - 1 - i] >> j) & 1;
            printf("%d", bit);
        }
        // Optional: add a space between bytes for readability
        if (i < size - 1) {
            printf(" ");
        }
    }
    printf("\n");
}


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

bf16_t bfpow16(bf16_t x, int n) {
    bf16_t temp = x;
    n -= 1;
    while(n--) temp = bfmul16(temp, x);
    return temp;
}

float floatfact(int n) {
    float temp = 1;
    for(int i = 1;i <= n;i++) temp = temp * (float)i;
    return temp;
}


uint16_t idiv7(int16_t a, int16_t b) {
    uint16_t r = 0;
    for (int i = 0; i < 16; i++) {
        a -= b;
        r = (r << 1) | (a >= 0);
        a = (a + (b & -(a < 0))) << 1;
    }

    return r;
}



bf16_t bfdiv16(bf16_t a, bf16_t b)
{
    uint16_t ia = a.bits;
    uint16_t ib = b.bits;

    /* divisor = 0 then return infty */
    if ((ib & 0x7FFF) == 0) return (bf16_t) {.bits = ((ia ^ ib) & 0x8000) | 0x7F80};
    /* dividend = 0 then return 0 */
    if ((ia & 0x7FFF) == 0) return (bf16_t) {.bits = (ia ^ ib) & 0x8000};

    /* mantissa */
    int16_t ma = (ia & 0x7F) | 0x80;
    int16_t mb = (ib & 0x7F) | 0x80;

    /* sign and exponent */
    int16_t sea = ia & 0xFF80;
    int16_t seb = ib & 0xFF80;

    /* result of mantissa */
    uint16_t mantissa = idiv7(ma, mb);
    int32_t mshift = !getbit(mantissa, 15);
    mantissa <<= mshift;

    return (bf16_t){.bits = ((sea - seb + 0x3F80) - (0x8000 & -mshift)) | (mantissa & 0x7E00) >> 9};
}


/* swap macro for uint32_t */
#define iswap(a, b) do { uint16_t temp = (a); (a) = (b); (b) = temp; } while (0)

int my_clz(uint16_t x) {
    int count = 0;
    uint16_t mask = 0x8000;  /* start with the highest bit set */

    while ((x & mask) == 0) {
        count++;
        mask >>= 1;  /* Shift the mask to the right */
    }

    return count;
}

bf16_t bfadd16(bf16_t a, bf16_t b) {
    uint16_t ia = a.bits;
    uint16_t ib = b.bits;

    /* compare thr absolute value */
    /* ensure a is bigget than b */
    uint16_t cmp_a = ia & 0x7fff;
    uint16_t cmp_b = ib & 0x7fff;
    if (cmp_a < cmp_b)
        iswap(ia, ib);
    
    /* exponent */
    uint16_t ea = (ia >> 7) & 0xff;
    uint16_t eb = (ib >> 7) & 0xff;

    /* 7 bits is the size of bf16_t mantissa */
    int16_t mantissa_align_shift = (ea - eb > 7) ? 7 : ea - eb;

    /* mantissa */
    uint16_t ma = (ia & 0x7F) | 0x80;
    uint16_t mb = (ib & 0x7F) | 0x80;

    mb >>= mantissa_align_shift;
    if((ia ^ ib) >> 15) ma -= mb;
    else ma += mb;
    int16_t clz = my_clz(ma);
    int16_t shift = 0;
    if(clz <= 8) {
        shift = 8 - clz;
        ma >>= shift;
        ea += shift;
    } else {
        shift = clz - 8;
        ma <<= shift;
        ea -= shift;
    }
    
    return (bf16_t){.bits = (ia & 0x8000) | (ea << 7) | ma};
}

bf16_t bfsub16(bf16_t a, bf16_t b) {
    uint16_t ia = a.bits;
    uint16_t ib = b.bits;

    /* compare the absolute values */
    /* ensure a has greater or equal magnitude than b */
    uint16_t cmp_a = ia & 0x7FFF;
    uint16_t cmp_b = ib & 0x7FFF;
    int swapped = 0;
    if (cmp_a < cmp_b) {
        iswap(ia, ib);
        swapped = 1;
    }

    /* exponents */
    uint16_t ea = (ia >> 7) & 0xFF;
    uint16_t eb = (ib >> 7) & 0xFF;

    /* 7 bits is the size of bf16_t mantissa */
    int16_t mantissa_align_shift = (ea - eb > 7) ? 7 : ea - eb;

    /* mantissas */
    int16_t ma = (ia & 0x7F) | 0x80;
    int16_t mb = (ib & 0x7F) | 0x80;

    mb >>= mantissa_align_shift;

    /* adjust mantissas based on signs */
    if (!((ia ^ ib) >> 15)) ma -= mb;
    else ma += mb;

    /* handle negative mantissa result */
    int negative_result = 0;
    if (ma < 0) {
        ma = -ma;
        negative_result = 1;
    }

    /* normalize the result */
    int16_t clz = my_clz(ma);
    int16_t shift = 0;
    if (clz <= 8) {
        shift = 8 - clz;
        ma >>= shift;
        ea += shift;
    } else {
        shift = clz - 8;
        ma <<= shift;
        ea -= shift;
    }

    /* determine the sign of the result */
    uint16_t sign_a = ia & 0x8000;
    uint16_t sign_b = ib & 0x8000;
    uint16_t sign;
    if (negative_result == 0) sign = swapped ? sign_b : sign_a;
    else sign = (swapped ? sign_b : sign_a) ^ 0x8000; /* Flip the sign */

    /* assemble the final result */
    return (bf16_t){.bits = sign | (ea << 7) | (ma & 0x7F)};
}


/* STL sigmoid approach */
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* random sample float */
float *sample(int sample_size) {
    float *arr = (float *)malloc(sample_size * sizeof(float));
    if (arr == NULL) {
        return NULL;
    }

    for (int i = 0; i < sample_size; i++) 
        arr[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    return arr;
}
