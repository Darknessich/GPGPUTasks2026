#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* values,
    __global       uint* counts,
    unsigned int n,
    unsigned int shift,
    unsigned int num_groups)
{
    __local uint hist[RADIX];

    const unsigned int local_id = get_local_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int block_start = group_id * RADIX_SORT_BLOCK_SIZE;

    if (local_id < RADIX)
        hist[local_id] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int k = 0; k < RADIX_SORT_VALUES_PER_THREAD; ++k) {
        const unsigned int i = block_start + k * GROUP_SIZE + local_id;
        if (i < n)
            atomic_inc(&hist[(values[i] >> shift) & RADIX_MASK]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < RADIX)
        counts[local_id * num_groups + group_id] = hist[local_id];
}
