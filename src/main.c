#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bf16.h"
#include "conversion.h"
#include "bf16utils.h"
#include "bfdiv16.h"
#include "bfmul16.h"
#include "bfadd16.h"


#define SAMPLE_SIZE 50
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

float MSE(float *arr1, float *arr2) {
	float mse = 0;
	for(int i = 0;i < SAMPLE_SIZE;i++)
		mse += powf(arr1[i] - arr2[i], 2);
	
	return mse / SAMPLE_SIZE;
}

void testMul(float a, float b) {
	printf("a = %f, b = %f\n", a, b);
    bf16_t bf_a = fp32_to_bf16(a), bf_b = fp32_to_bf16(b);
    bf16_t bfmul = bfmul16(bf_a, bf_b);
    float mul_result = bf16_to_fp32(bfmul);
    printf("FP32 to BF16 multiplication %f * %f = %f\n", a, b, mul_result);
    printf("bfmul bit pattern : ");
    printBit(&bfmul.bits, sizeof(bfmul.bits));
    printf("mul_result bit pattern : ");
    printBit(&mul_result, sizeof(mul_result));
}

void testDiv(float a, float b) {
    bf16_t bf_a = fp32_to_bf16(a), bf_b = fp32_to_bf16(b);
    bf16_t bfdiv = bfdiv16(bf_a, bf_b);
    float div_result = bf16_to_fp32(bfdiv);
    printf("FP32 to BF16 division %f / %f = %f\n", a, b, div_result);
    printf("bfdiv bit pattern : ");
    printBit(&bfdiv.bits, sizeof(bfdiv.bits));
    printf("div_result bit pattern : ");
    printBit(&div_result, sizeof(div_result));
}

void testAdd(float a, float b) {
    bf16_t bf_a = fp32_to_bf16(a), bf_b = fp32_to_bf16(b);
    bf16_t bfadd = bfadd16(bf_a, bf_b);
    float add_result = bf16_to_fp32(bfadd);
    printf("FP32 to BF16 addition %f + %f = %f\n", a, b, add_result);
    printf("bfadd bit pattern : ");
    printBit(&bfadd.bits, sizeof(bfadd.bits));
    printf("add_result bit pattern : ");
    printBit(&add_result, sizeof(add_result));
}

void testSub(float a, float b) {
    bf16_t bf_a = fp32_to_bf16(a), bf_b = fp32_to_bf16(b);
	bf16_t bfsub = bfsub16(bf_a, bf_b);
	float sub_result = bf16_to_fp32(bfsub);
    printf("FP32 to BF16 subtraction %f - %f = %f\n", a, b, sub_result);
    printf("bfsub bit pattern : ");
    printBit(&bfsub.bits, sizeof(bfsub.bits));
    printf("sub_result bit pattern : ");
    printBit(&sub_result, sizeof(sub_result));
}

void testBF16MulMSE(float *arr1, float *arr2) {
	float BF16Mul[SAMPLE_SIZE], FP32Mul[SAMPLE_SIZE];
	for(int i = 0;i != SAMPLE_SIZE;i++) {
		BF16Mul[i] = bf16_to_fp32(bfmul16(fp32_to_bf16(arr1[i]), fp32_to_bf16(arr2[i])));
		FP32Mul[i] = arr1[i] * arr2[i];
	}
	float mse = MSE(BF16Mul, FP32Mul);
	printf("bfmul16 MSE : %f\n", mse);
}

void testBF16AddMSE(float *arr1, float *arr2) {
	float BF16Add[SAMPLE_SIZE], FP32Add[SAMPLE_SIZE];
	for(int i = 0;i != SAMPLE_SIZE;i++) {
		BF16Add[i] = bf16_to_fp32(bfadd16(fp32_to_bf16(arr1[i]), fp32_to_bf16(arr2[i])));
		FP32Add[i] = arr1[i] + arr2[i];
	}
	float mse = MSE(BF16Add, FP32Add);
	printf("bfadd16 MSE : %f\n", mse);
}

void testBF16SubMSE(float *arr1, float *arr2) {
	float BF16Sub[SAMPLE_SIZE], FP32Sub[SAMPLE_SIZE];
	for(int i = 0;i != SAMPLE_SIZE;i++) {
		BF16Sub[i] = bf16_to_fp32(bfsub16(fp32_to_bf16(arr1[i]), fp32_to_bf16(arr2[i])));
		FP32Sub[i] = arr1[i] - arr2[i];
	}
	float mse = MSE(BF16Sub, FP32Sub);
	printf("bfsub16 MSE : %f\n", mse);
}

int main() {
	float *arr1 = sample(SAMPLE_SIZE, -5, 5);
	float *arr2 = sample(SAMPLE_SIZE, -5, 5);
	testBF16MulMSE(arr1, arr2);
	testBF16AddMSE(arr1, arr2);
	testBF16SubMSE(arr1, arr2);
	free(arr1);
	free(arr2);
}
