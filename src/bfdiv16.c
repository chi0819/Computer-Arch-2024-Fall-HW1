#include <stdint.h>
#include "bf16.h"
#include "bf16utils.h"

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
