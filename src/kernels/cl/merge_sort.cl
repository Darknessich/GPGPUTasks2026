#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define VT    MERGE_SORT_VALUES_PER_THREAD
#define BLOCK MERGE_SORT_BLOCK_SIZE

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(i) ((i) >> LOG_NUM_BANKS)
#define PADDED(i) ((i) + CONFLICT_FREE_OFFSET(i))

int merge_path_global(__global const uint* a, int a_n, __global const uint* b, int b_n, int diag)
{
    int lo = max(0, diag - b_n) - 1;
    int hi = min(diag, a_n);
    while (lo < hi - 1) {
        const int mid = (lo + hi) / 2;
        if (a[mid] <= b[diag - 1 - mid])
            lo = mid;
        else
            hi = mid;
    }
    return hi;
}

int merge_path_local(__local const uint* block, int a_start, int a_n, int b_start, int b_n, int diag)
{
    int lo = max(0, diag - b_n) - 1;
    int hi = min(diag, a_n);
    while (lo < hi - 1) {
        const int mid = (lo + hi) / 2;
        if (block[PADDED(a_start + mid)] <= block[PADDED(b_start + diag - 1 - mid)])
            lo = mid;
        else
            hi = mid;
    }
    return hi;
}

void merge_local(
    __local const uint* block,
    int a_start, int a_n,
    int b_start, int b_n,
    int i, int j, int count, uint* values)
{
    for (int k = 0; k < VT; ++k) {
        if (k < count) {
            if (i < a_n && (j >= b_n || block[PADDED(a_start + i)] <= block[PADDED(b_start + j)])) {
                values[k] = block[PADDED(a_start + i)];
                ++i;
            } else {
                values[k] = block[PADDED(b_start + j)];
                ++j;
            }
        }
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort_01_local(
    __global const uint* input_data,
    __global       uint* output_data,
    unsigned int n)
{
    __local uint block[PADDED(BLOCK - 1) + 1];

    const unsigned int local_id = get_local_id(0);
    const unsigned int block_start = get_group_id(0) * BLOCK;

    for (unsigned int k = 0; k < VT; ++k) {
        const unsigned int i = k * GROUP_SIZE + local_id;
        block[PADDED(i)] = (block_start + i < n) ? input_data[block_start + i] : UINT_MAX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint values[VT];
    for (int k = 0; k < VT; ++k)
        values[k] = block[PADDED(local_id * VT + k)];

    // Сортируем VT значений с помощью [odd-even sort](https://en.wikipedia.org/wiki/Odd%E2%80%93even_sort)
    // Не вставками, так как odd-even лучше оптимизируется для GPU
    // Не merge sort, так как пришлось бы усложнять логику программы (следующего цикла)
    for (int round = 0; round < VT; ++round) {
        for (int k = round & 1; k + 1 < VT; k += 2) {
            const uint lo = min(values[k], values[k + 1]);
            const uint hi = max(values[k], values[k + 1]);
            values[k] = lo;
            values[k + 1] = hi;
        }
    }

    for (int w = VT; w < BLOCK; w *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < VT; ++k)
            block[PADDED(local_id * VT + k)] = values[k];
        barrier(CLK_LOCAL_MEM_FENCE);

        const int out = local_id * VT;
        const int pair_start = out & ~(2 * w - 1);
        const int diag = out - pair_start;
        const int from_a = merge_path_local(block, pair_start, w, pair_start + w, w, diag);
        merge_local(block, pair_start, w, pair_start + w, w, from_a, diag - from_a, VT, values);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < VT; ++k)
        block[PADDED(local_id * VT + k)] = values[k];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int k = 0; k < VT; ++k) {
        const unsigned int i = k * GROUP_SIZE + local_id;
        if (block_start + i < n)
            output_data[block_start + i] = block[PADDED(i)];
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort_02_partition(
    __global const uint* input_data,
    __global       uint* splits,
    unsigned int sorted_k,
    unsigned int n,
    unsigned int num_blocks)
{
    const unsigned int block_id = get_global_id(0);
    if (block_id >= num_blocks)
        return;

    const unsigned int pos = block_id * BLOCK;
    const unsigned int pair_start = pos & ~(2 * sorted_k - 1);
    const unsigned int a_n = min(sorted_k, n - pair_start);
    const unsigned int b_n = min(sorted_k, n - pair_start - a_n);

    __global const uint* a = input_data + pair_start;
    __global const uint* b = a + sorted_k;
    splits[block_id] = merge_path_global(a, a_n, b, b_n, pos - pair_start);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort_03_merge(
    __global const uint* input_data,
    __global const uint* splits,
    __global       uint* output_data,
    unsigned int sorted_k,
    unsigned int n)
{
    __local uint block[PADDED(BLOCK - 1) + 1];

    const unsigned int local_id = get_local_id(0);
    const unsigned int block_id = get_group_id(0);
    const unsigned int pos = block_id * BLOCK;
    const unsigned int pair_start = pos & ~(2 * sorted_k - 1);
    const unsigned int a_n = min(sorted_k, n - pair_start);
    const unsigned int b_n = min(sorted_k, n - pair_start - a_n);

    const unsigned int diag_begin = pos - pair_start;
    const unsigned int diag_end = min(diag_begin + BLOCK, a_n + b_n);
    const unsigned int a_begin = splits[block_id];
    const unsigned int a_end = (diag_end == a_n + b_n) ? a_n : splits[block_id + 1];
    const unsigned int b_begin = diag_begin - a_begin;
    const int block_n = diag_end - diag_begin;
    const int block_a_n = a_end - a_begin;
    const int block_b_n = block_n - block_a_n;

    __global const uint* a = input_data + pair_start + a_begin;
    __global const uint* b = input_data + pair_start + sorted_k + b_begin;
    for (int k = 0; k < VT; ++k) {
        const int i = k * GROUP_SIZE + local_id;
        if (i < block_n)
            block[PADDED(i)] = (i < block_a_n) ? a[i] : b[i - block_a_n];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int out = min((int) (local_id * VT), block_n);
    const int from_a = merge_path_local(block, 0, block_a_n, block_a_n, block_b_n, out);
    uint values[VT];
    merge_local(block, 0, block_a_n, block_a_n, block_b_n, from_a, out - from_a, block_n - out, values);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < VT; ++k) {
        if (out + k < block_n)
            block[PADDED(out + k)] = values[k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < VT; ++k) {
        const int i = k * GROUP_SIZE + local_id;
        if (i < block_n)
            output_data[pos + i] = block[PADDED(i)];
    }
}
