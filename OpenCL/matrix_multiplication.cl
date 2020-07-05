#define TILE_W 16
#define TILE_H 16

__kernel void buffered_matrix_multiplication(
        __global const float *first,
        __global const float *second,
        __global float *result,
        const uint n,
        const uint m,
        const uint k
) {
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint tile_x = get_local_id(0);
    const uint tile_y = get_local_id(1);

    __local float first_buffer[TILE_W][TILE_H];
    __local float second_buffer[TILE_W][TILE_H];

    float sum = 0;
    for (int i = 0; i < m; i += TILE_W) {
        first_buffer[x][y] = first[tile_x * m + tile_y];
        second_buffer[x][y] = second[tile_y * k + tile_x];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < TILE_H; ++j) {
            sum += first_buffer[x][j] * second_buffer[j][y];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    result[tile_x * n + tile_y] = sum;
}

#define WORK_PER_THREAD 4

__kernel void reduced_matrix_multiplication(
        __global const float *first,
        __global const float *second,
        __global float *result,
        const uint n,
        const uint m,
        const uint k
) {
    const uint x = get_global_id(1) * WORK_PER_THREAD;
    const uint y = get_global_id(0);
    const uint tile_x = get_local_id(1) * WORK_PER_THREAD;
    const uint tile_y = get_local_id(0);

    __local float first_buffer[TILE_W][TILE_H];
    __local float second_buffer[TILE_W][TILE_H];

    float sum[WORK_PER_THREAD] = { 0 };
    for (uint i = 0; i < m; i += TILE_W) {
        for (uint work = 0; work < WORK_PER_THREAD; ++work) {
            const uint row = i + tile_x + work;
            const uint col = i + tile_y;

            first_buffer[tile_x + work][tile_y] = first[(x + work) * m + col];
            second_buffer[tile_x + work][tile_y] = second[row * k + y];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint work = 0; work < WORK_PER_THREAD; ++work) {
            for (uint t = 0; t < TILE_H; ++t) {
                sum[work] += first_buffer[tile_x + work][t] * second_buffer[t][tile_y];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint work = 0; work < WORK_PER_THREAD; ++work) {
        result[(x + work) * k + y] = sum[work];
    }
}