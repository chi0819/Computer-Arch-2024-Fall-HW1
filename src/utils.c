#include <stdlib.h>
#include <math.h>
#include "bf16.h"

// bf16_t bfmul16(bf16_t a, bf16_t b) {
//     // TODO
// }
//
// bf16_t bfdiv16(bf16_t a, bf16_t b) {
//     // TODO
// }
//
// bf16_t bfadd16(bf16_t a, bf16_t b) {
//     // TODO
// }

// STL Sigmoid approach
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// random sample float to test STL Sigmoid approach and BF16 without floating point support Sigmoid approach
float *sample(int sample_size) {
    float *arr = (float *)malloc(sample_size * sizeof(float));
    if (arr == NULL) {
        return NULL;
    }

    for (int i = 0; i < sample_size; i++) 
        arr[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    return arr;
}
