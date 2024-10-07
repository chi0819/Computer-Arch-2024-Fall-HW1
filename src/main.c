#include <stdio.h>

#include "bf16.h"
#include "conversion.h"
#include "utils.h"

#define SAMPLE_SIZE 10

int main() {
    float *arr = sample(SAMPLE_SIZE);
    float output[SAMPLE_SIZE] = {};

    for(int i = 0;i <= SAMPLE_SIZE;i++)
        output[i] = sigmoid(arr[i]);

    for(int i = 0;i <= SAMPLE_SIZE;i++)
        printf("x = %f, sigmoid(x) = %f\n", arr[i], output[i]);

    return 0;
}
