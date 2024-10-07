# Sigmoid Function by BF16 without Floating Point Support
## C Code Structure
```
.
├── Makefile
├── README.md
├── include
│   ├── type
│   │   ├── bf16.h
│   │   └── conversion.h
│   └── utils.h
└── src
    ├── main.c
    ├── type
    │   └── conversion.c
    └── utils.c
```
## type
- `bf16.h` : Include `bf16_t` data type
- `conversion.h` : Include function to convert `FP32` to `BF16` and `BF16` to `FP32`

## `utils`
In `utils` include function to randomly sample `float` in range $[-1,1]$  
Also has `FP32` Sigmoid function to test the accuracy of BF16 without floating point support Sigmoid  
The `FP32` Sigmoid function used `expf(x)` in STL  
Other function in `utils` is the implementations for `BF16` calculation  
- `fmul32` -> `bfmul16`
- `fdiv32` -> `bfdiv16`
- `fadd32` -> `bfadd16`

## `main`
In `main`, I try to compare the accuracy of STL version Sigmoid and BF16 Sigmoid without floating point support  
And the RISC-V assembly code to enhance performance will be update later  

## RISC-V Assembly Code
TODO
