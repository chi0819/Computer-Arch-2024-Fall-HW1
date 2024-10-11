#include <stdio.h>

#include "bf16.h"
#include "conversion.h"
#include "utils.h"

#define SAMPLE_SIZE 10

int main() {
    // float *arr = sample(SAMPLE_SIZE);
    // float output[SAMPLE_SIZE] = {};
    //
    // for(int i = 0;i <= SAMPLE_SIZE;i++)
    //     output[i] = sigmoid(arr[i]);
    //
    // for(int i = 0;i <= SAMPLE_SIZE;i++)
    //     printf("x = %f, sigmoid(x) = %f\n", arr[i], output[i]);
    //
    //free(arr);
    float a = 0.125, b = 6;
    bf16_t bf_a = fp32_to_bf16(a), bf_b = fp32_to_bf16(b);
    bf16_t bfmul = bfmul16(bf_a, bf_b);
    bf16_t bfdiv = bfdiv16(bf_a, bf_b);
    bf16_t bfadd = bfadd16(bf_a, bf_b);
    bf16_t bfsub = bfsub16(bf_a, bf_b);
    float mul_result = bf16_to_fp32(bfmul);
    float div_result = bf16_to_fp32(bfdiv);
    float add_result = bf16_to_fp32(bfadd);
    float sub_result = bf16_to_fp32(bfsub);

    printf("FP32 to BF16 multiplication %f * %f = %f\n", a, b, mul_result);
    printf("bfmul bit pattern : ");
    printBit(&bfmul.bits, sizeof(bfmul.bits));
    printf("mul_result bit pattern : ");
    printBit(&mul_result, sizeof(mul_result));

    printf("FP32 to BF16 division %f / %f = %f\n", a, b, div_result);
    printf("bfdiv bit pattern : ");
    printBit(&bfdiv.bits, sizeof(bfdiv.bits));
    printf("div_result bit pattern : ");
    printBit(&div_result, sizeof(div_result));

    printf("FP32 to BF16 addition %f + %f = %f\n", a, b, add_result);
    printf("bfadd bit pattern : ");
    printBit(&bfadd.bits, sizeof(bfadd.bits));
    printf("add_result bit pattern : ");
    printBit(&add_result, sizeof(add_result));

    printf("FP32 to BF16 subtraction %f - %f = %f\n", a, b, sub_result);
    printf("bfsub bit pattern : ");
    printBit(&bfsub.bits, sizeof(bfsub.bits));
    printf("sub_result bit pattern : ");
    printBit(&sub_result, sizeof(sub_result));

    printf("\n\n\n\n\n");

    int n = 10;
    float fp32_x  = 0.5;
    bf16_t x = fp32_to_bf16(fp32_x);

    /* 0x3F80 in bf16_t is '1', represent the first term in exp expansion */
    bf16_t my_sigmoid = {.bits = 0x3F80};
    for(int i = 1;i < n;i++) {
        bf16_t term = bfdiv16(bfpow16(x,i),fp32_to_bf16(floatfact(i)));
        printf("term %d : %f = %f / %f\n", i, bf16_to_fp32(term), bf16_to_fp32(bfpow16(x, i)), floatfact(i));
        if(i%2) my_sigmoid = bfsub16(my_sigmoid, term);
        else my_sigmoid = bfadd16(my_sigmoid, term);
    }

    my_sigmoid = bfdiv16((bf16_t){.bits = 0x3F80}, bfadd16((bf16_t){.bits = 0x3F80}, my_sigmoid));

    printf("bfpow16(x,2) = %f\n", bf16_to_fp32(bfpow16(x, 2)));
    printf("floatfact(3) = %f\n", floatfact(3));

    printf("input x = %f\n", fp32_x);
    printf("my bf16 sigmoid = %f\n", bf16_to_fp32(my_sigmoid));
    printf("STL sigmoid approch = %f\n", sigmoid(fp32_x));

    return 0;
}
