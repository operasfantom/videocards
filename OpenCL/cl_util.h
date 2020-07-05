#ifndef OPENCL_CL_UTIL_H
#define OPENCL_CL_UTIL_H

#include <CL/opencl.h>

struct working_params {
    cl_platform_id platform_id;

    cl_uint devices_count;
    cl_device_id *devices;
    cl_command_queue_properties command_queue_properties;

    cl_context context;

    cl_device_id device;
    cl_command_queue command_queue;
    cl_program program;

    cl_kernel kernel;
};

void obtain_platforms(cl_uint *platforms_count,
                      cl_platform_id **platform_ids);

void find_suitable_devices(cl_uint platforms_count,
                           cl_platform_id *platform_ids,
                           struct working_params *params);

void create_context(struct working_params *params);

void create_command_queue(struct working_params *params);

void create_program(struct working_params *params,
                    const char *kernel_file_name);

void build_program(struct working_params *params);

void print_profile_info(cl_event run_event,
                        float ops);

void create_kernel(struct working_params *params,
                   char *fun_name);

cl_mem create_buffer(struct working_params *params,
                     size_t size,
                     cl_mem_flags flags);

void enqueue_write_buffer(struct working_params *params,
                          cl_mem buffer,
                          size_t size,
                          const void *ptr);

void set_kernel_arg(struct working_params *params,
                    int index,
                    size_t arg_size,
                    const void *arg_value);


void enqueue_nd_range_kernel(struct working_params *params,
                             const size_t *global_work_size,
                             const size_t *local_work_size,
                             cl_event *run_event);

void enqueue_read_buffer(struct working_params *params,
                         cl_mem buffer,
                         size_t size,
                         void *ptr);


void run_algorithm(const char *file_name,
                   float ops,
                   cl_event (*algorithm)(struct working_params *));

#endif //OPENCL_CL_UTIL_H
