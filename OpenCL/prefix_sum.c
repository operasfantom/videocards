#include "prefix_sum.h"

cl_event buffered_prefix_sum(struct working_params *params) {
    cl_int size = L;

    cl_float *array = NULL;
    cl_float *result_array = NULL;

    size_t memory_size;
    generate_array(size, &memory_size, &array, &result_array);

    create_kernel(params, "prefix_sum");

    cl_mem buffer = create_buffer(params, memory_size, CL_MEM_READ_ONLY);
    cl_mem result_buffer = create_buffer(params, memory_size, CL_MEM_READ_WRITE);

    enqueue_write_buffer(params, buffer, memory_size, array);

    set_kernel_arg(params, 0, sizeof(cl_mem), &buffer);
    set_kernel_arg(params, 1, sizeof(cl_mem), &result_buffer);
    set_kernel_arg(params, 2, sizeof(cl_uint), &size);

    const size_t global_work_size[] = {size / 2};
    const size_t local_work_size[] = {size / 2};

    cl_event run_event;
    enqueue_nd_range_kernel(params, global_work_size, local_work_size, &run_event);
    enqueue_read_buffer(params, result_buffer, memory_size, result_array);

    verify_prefix_sum(size, array, result_array);
    return run_event;
}
