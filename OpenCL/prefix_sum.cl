__kernel void prefix_sum(__global const float *array, __global float *result, uint n) {
    int id = get_local_id(0);

    __local float buffer[256];

    buffer[2 * id] = array[2 * id];
    buffer[2 * id + 1] = array[2 * id + 1];

    int offset = 1;
    for (int d = n / 2; d > 0; d /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id < d) {
            int i = offset * (2 * id + 1) - 1;
            int j = offset * (2 * id + 2) - 1;
            buffer[j] += buffer[i];
        }
        offset *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (id == 0) {
        buffer[n - 1] = 0;
    }

    for (int d = 1; d < n; d *= 2) {
        offset /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id < d) {
            int i = offset * (2 * id + 1) - 1;
            int j = offset * (2 * id + 2) - 1;
            float t = buffer[i];
            buffer[i] = buffer[j];
            buffer[j] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    result[2 * id] = buffer[2 * id] + array[2 * id];
    result[2 * id + 1] = buffer[2 * id + 1] + array[2 * id + 1];
}