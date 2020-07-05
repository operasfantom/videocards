#include "matrix_multiplication.h"

#include "matrix_util.h"
#include "constants.h"


cl_event buffered_matrix_multiplication(struct working_params *params) {
    cl_uint first_size = N;
    cl_uint second_size = M;
    cl_uint third_size = K;

    cl_float *first_matrix = NULL;
    cl_float *second_matrix = NULL;
    cl_float *result_matrix = NULL;
    size_t first_memory_size, second_memory_size, result_memory_size;
    generate_matrix(first_size, second_size, third_size,
                    &first_memory_size, &second_memory_size, &result_memory_size,
                    &first_matrix, &second_matrix, &result_matrix);

    create_kernel(params, "reduced_matrix_multiplication");

    cl_mem first_buffer = create_buffer(params, first_memory_size, CL_MEM_READ_ONLY);
    cl_mem second_buffer = create_buffer(params, second_memory_size, CL_MEM_READ_ONLY);
    cl_mem result_buffer = create_buffer(params, result_memory_size, CL_MEM_READ_WRITE);

    enqueue_write_buffer(params, first_buffer, first_memory_size, first_matrix);
    enqueue_write_buffer(params, second_buffer, second_memory_size, second_matrix);

    set_kernel_arg(params, 0, sizeof(cl_mem), &first_buffer);
    set_kernel_arg(params, 1, sizeof(cl_mem), &second_buffer);
    set_kernel_arg(params, 2, sizeof(cl_mem), &result_buffer);
    set_kernel_arg(params, 3, sizeof(cl_uint), &first_size);
    set_kernel_arg(params, 4, sizeof(cl_uint), &second_size);
    set_kernel_arg(params, 5, sizeof(cl_uint), &third_size);

    const size_t global_work_size[] = {third_size, first_size / WORK_PER_THREAD};
    const size_t local_work_size[] = {TILE_W, TILE_H / WORK_PER_THREAD};

    cl_event run_event;
    enqueue_nd_range_kernel(params, global_work_size, local_work_size, &run_event);
    enqueue_read_buffer(params, result_buffer, result_memory_size, result_matrix);

    verify_matrix_multiplication(first_size, second_size, third_size,
                                 first_matrix, second_matrix, result_matrix);
    return run_event;
}
