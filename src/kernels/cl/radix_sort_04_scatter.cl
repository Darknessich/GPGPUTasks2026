#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(i) ((i) >> LOG_NUM_BANKS)
#define PADDED(i) ((i) + CONFLICT_FREE_OFFSET(i))

#define HIST_SIZE (RADIX * GROUP_SIZE)

uint work_group_scan(uint value, __local uint* temp)
{
    const unsigned int local_id = get_local_id(0);
    temp[local_id] = value;
    for (unsigned int offset = 1; offset < GROUP_SIZE; offset <<= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        const uint add = (local_id >= offset) ? temp[local_id - offset] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        temp[local_id] += add;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return temp[local_id] - value;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* values,
    __global const uint* prefix_sums,
    __global       uint* output,
    unsigned int n,
    unsigned int shift,
    unsigned int num_groups)
{
    __local uint block_values[PADDED(RADIX_SORT_BLOCK_SIZE - 1) + 1];
    __local ushort hist[PADDED(HIST_SIZE - 1) + 1];
    __local uint scan_temp[GROUP_SIZE];
    __local uint global_digit_start[RADIX];
    __local uint local_digit_start[RADIX];

    const unsigned int local_id = get_local_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int block_start = group_id * RADIX_SORT_BLOCK_SIZE;
    const unsigned int block_n = min((unsigned int) RADIX_SORT_BLOCK_SIZE, n - block_start);

    if (local_id < RADIX) {
        const unsigned int i = local_id * num_groups + group_id;
        global_digit_start[local_id] = (i == 0) ? 0 : prefix_sums[i - 1];
    }
    for (unsigned int k = 0; k < RADIX_SORT_VALUES_PER_THREAD; ++k) {
        const unsigned int i = k * GROUP_SIZE + local_id;
        if (i < block_n)
            block_values[PADDED(i)] = values[block_start + i];
    }
    for (unsigned int d = 0; d < RADIX; ++d)
        hist[PADDED(d * GROUP_SIZE + local_id)] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint my_values[RADIX_SORT_VALUES_PER_THREAD];
    uint my_ranks[RADIX_SORT_VALUES_PER_THREAD];
    for (unsigned int k = 0; k < RADIX_SORT_VALUES_PER_THREAD; ++k) {
        const unsigned int i = local_id * RADIX_SORT_VALUES_PER_THREAD + k;
        if (i < block_n) {
            const uint value = block_values[PADDED(i)];
            const unsigned int h = PADDED(((value >> shift) & RADIX_MASK) * GROUP_SIZE + local_id);
            my_values[k] = value;
            my_ranks[k] = hist[h];
            hist[h] = my_ranks[k] + 1;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

  const unsigned int chunk_start = local_id * RADIX;
    uint chunk_sum = 0;
    for (unsigned int j = 0; j < RADIX; ++j) {
        const unsigned int h = PADDED(chunk_start + j);
        const uint count = hist[h];
        hist[h] = chunk_sum;
        chunk_sum += count;
    }
    const uint chunk_offset = work_group_scan(chunk_sum, scan_temp);
    for (unsigned int j = 0; j < RADIX; ++j)
        hist[PADDED(chunk_start + j)] += chunk_offset;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < RADIX)
        local_digit_start[local_id] = hist[PADDED(local_id * GROUP_SIZE)];
    for (unsigned int k = 0; k < RADIX_SORT_VALUES_PER_THREAD; ++k) {
        if (local_id * RADIX_SORT_VALUES_PER_THREAD + k < block_n) {
            const unsigned int h = PADDED(((my_values[k] >> shift) & RADIX_MASK) * GROUP_SIZE + local_id);
            block_values[PADDED(hist[h] + my_ranks[k])] = my_values[k];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int k = 0; k < RADIX_SORT_VALUES_PER_THREAD; ++k) {
        const unsigned int i = k * GROUP_SIZE + local_id;
        if (i < block_n) {
            const uint value = block_values[PADDED(i)];
            const unsigned int digit = (value >> shift) & RADIX_MASK;
            output[global_digit_start[digit] + i - local_digit_start[digit]] = value;
        }
    }
}
