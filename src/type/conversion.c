#include "conversion.h"

bf16_t fp32_to_bf16(float s)
{
    bf16_t h;
    union {
        float f;
        uint32_t i;
    } u = {.f = s};

    if ((u.i & 0x7FFFFFFF) > 0x7F800000) { /* NaN */
        h.bits = (u.i >> 16) | 0x40;         /* Force to quiet NaN */
        return h;
    }

    h.bits = (u.i + (0x7FFF + ((u.i >> 16) & 1))) >> 16;
    return h;
}

float bf16_to_fp32(bf16_t h)
{
    union {
        float f;
        uint32_t i;
    } u = {.i = ((uint32_t)h.bits) << 16};
    return u.f;
}
