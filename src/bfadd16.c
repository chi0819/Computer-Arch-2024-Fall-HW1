#include <stdint.h>
#include "bf16.h"
#include "bf16utils.h"

#define iswap(a, b) do { uint16_t temp = (a); (a) = (b); (b) = temp; } while (0)

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
    
    return (bf16_t){.bits = (ia & 0x8000) | ((ea << 7) & 0x7f80) | (ma & 0x7f)};
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
	/* if sign is same then ma subtract mb directly */
    if (!((ia ^ ib) >> 15)) ma -= mb;
    else ma += mb;

    /* handle negative mantissa result */
    int negative_result = 0;
    if (ma < 0) {
        ma = -ma;
        negative_result = 1;
    }

    /* normalize the result */
    int16_t clz = my_clz((uint16_t)ma);
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
    uint16_t sign_a = ia & 0x8000; /* if swapped ia = original ib */
    uint16_t sign_b = ib & 0x8000; /* if swapped ib = original ia */
    uint16_t sign;
    if (negative_result == 0) sign = (swapped ? sign_b : sign_a);
    else sign = (swapped ? sign_b : sign_a) ^ 0x8000; /* Flip the sign */
		
	if(swapped && !(sign_a ^ sign_b)) sign ^= 0x8000;

    /* assemble the final result */
    return (bf16_t){.bits = sign | ((ea << 7) & 0x7f80) | (ma & 0x7F)};
}
