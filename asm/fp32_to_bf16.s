.data
argument: .word   0x3E800000 # float 0.25

.text

_start:
    j main
    
fp32_to_bf16:
    la a0, argument      # load argument address
    lw a1, 0(a0)         # load data (float 0.25) from argument
    li a2, 0x7F800000    # use to check NaN case
    li a3, 0x7FFFFFF     # bit mask
    and a4, a1, a3       # u.i & 0x7FFFFFFF
    blt a2, a4, non_case # Handle NaN case
    li a2, 0x7FFF
    mv a3, a1
    srli a3, a3, 16      # u.i >> 16
    andi a3, a3, 1       # (u.i >> 16) & 1
    add a3, a3, a2       # 0x7FFF + (u.i >> 16) & 1
    add a3, a3, a1       # u.i + (0x7FFF + (u.i >> 16) & 1)
    srli a0, a3, 16      # (u.i + (0x7FFF + (u.i >> 16) & 1)) >> 16
    ret
non_case:
    srli a1, a1, 16      # (u.i >> 16)
    li a4, 0x40
    or a0, a1, a4        # (u.i >> 16) | 0x40
    ret
main:
    jal ra, fp32_to_bf16
    
end:
    nop
