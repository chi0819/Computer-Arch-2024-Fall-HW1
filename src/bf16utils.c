#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "bf16.h"
#include "bfmul16.h"

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
