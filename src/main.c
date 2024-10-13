#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bf16.h"
#include "conversion.h"
#include "bf16utils.h"
#include "bfdiv16.h"
#include "bfmul16.h"
#include "bfadd16.h"


#define SAMPLE_SIZE 10
#define ITERATION 5

/* 
 * Convert float x to bf16_t
 * After calculation will extend bf16_t back to float
 */
float my_sigmoid(float x) {
    /* 0x3F80 in bf16_t is '1', represent the first term in exp expansion */
    bf16_t term = {.bits = 0x3F80};
    for(int i = 1;i < ITERATION;i++) {
        bf16_t temp_term = bfdiv16(bfpow16(fp32_to_bf16(x),i),fp32_to_bf16(floatfact(i))); /* (x^i / i!) */
        if(i%2) term = bfsub16(term, temp_term);
        else term = bfadd16(term, temp_term);
    }
	bf16_t ans = bfdiv16(fp32_to_bf16(1), bfadd16(fp32_to_bf16(1), term));
	return bf16_to_fp32(ans);
}

/* Calculate the Mean Square Error for bf16_t my_sigmoid */
float MSE(float *arr) {
	float mse = 0;
	for(int i = 0;i <= SAMPLE_SIZE;i++)
		mse += powf(sigmoid(arr[i]) - my_sigmoid(arr[i]), 2);
	
	return mse / SAMPLE_SIZE;
}

/* Used to test each function working well */
void test(float a, float b) {
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
}

int main() {
	float *arr = sample(SAMPLE_SIZE);
	float mse = MSE(arr);
	printf("MSE : %f\n", mse);
}
