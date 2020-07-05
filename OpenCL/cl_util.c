#include <string.h>
#include "matrix_multiplication.h"
#include "cl_util.h"

#include "constants.h"

#include <stdio.h>
#include <stdlib.h>

#ifndef CHECK_ERR
#define CHECK_ERR(intro, result, action)                      \
{                                                             \
    if ((result) < 0)                                         \
    {                                                         \
        fprintf(stderr, "%s! Error code: %d", intro, result); \
        action;                                               \
    }                                                         \
}
#endif

#define DTILE_W "16"
#define DTILE_H "16"

void print_platform_info(const cl_platform_id platform_id) {
    cl_platform_info info_types[3] = {CL_PLATFORM_VERSION, CL_PLATFORM_NAME, CL_PLATFORM_VENDOR};
    for (int i = 0; i < sizeof info_types / sizeof info_types[0]; ++i) {
        size_t buffer_size;
        clGetPlatformInfo(platform_id, info_types[i], 0, NULL, &buffer_size);

        char *response_buffer = (char *) malloc(buffer_size * sizeof(char));
        clGetPlatformInfo(platform_id, info_types[i], buffer_size, response_buffer, NULL);

        printf("%s\t\t", response_buffer);
        free(response_buffer);
    }
    printf("\n");
}

void print_device_info(cl_device_id device_id) {
    cl_device_info info_types[3] = {CL_DEVICE_AVAILABLE, CL_DEVICE_COMPILER_AVAILABLE, CL_DEVICE_HOST_UNIFIED_MEMORY};
    for (int i = 0; i < 3; ++i) {
        size_t buffer_size;
        clGetDeviceInfo(device_id, info_types[i], 0, NULL, &buffer_size);

        char *response_buffer = (char *) malloc(buffer_size * sizeof(char));
        clGetDeviceInfo(device_id, info_types[i], buffer_size, response_buffer, NULL);

        if (buffer_size == sizeof(int)) {
            printf("%d\t\t", *(int *) response_buffer);
        } else {
            printf("--unknown--\t\t");
        }

        free(response_buffer);
    }
    printf("\n");
}

void filter_devices(cl_uint *devices_count,
                    cl_device_id *devices) {
    int last_unfit = 0;
    for (int i = 0; i < *devices_count; ++i) {
        size_t buffer_size;
        const cl_device_id device_id = devices[i];
        clGetDeviceInfo(device_id, CL_DEVICE_HOST_UNIFIED_MEMORY, 0, NULL, &buffer_size);

        char *response_buffer = (char *) malloc(buffer_size * sizeof(char));
        clGetDeviceInfo(device_id, CL_DEVICE_HOST_UNIFIED_MEMORY, buffer_size, response_buffer, NULL);

        const int unified = *(int *) response_buffer;
        if (!unified) {
            devices[last_unfit++] = device_id;
        }
    }
    *devices_count = last_unfit;
}

void printf_build_error(struct working_params const *params) {
    char *error_buf = (char *) malloc(2048 * sizeof(char));
    const cl_int result = clGetProgramBuildInfo(params->program, params->device, CL_PROGRAM_BUILD_LOG, 2048,
                                                error_buf, NULL);
    CHECK_ERR("printf_build_error::clGetProgramBuildInfo", result, exit(0))

    printf("%s", error_buf);

    free(error_buf);
    exit(0);
}

void print_profile_info(cl_event run_event, float ops) {
    cl_ulong t_start = 0, t_end = 0;

    cl_int result = clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_START, sizeof t_start, &t_start, 0);
    CHECK_ERR("print_profile_info::clGetEventProfilingInfo", result, exit(0));

    result = clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_END, sizeof t_end, &t_end, 0);
    CHECK_ERR("print_profile_info::clGetEventProfilingInfo", result, exit(0));

    cl_ulong ns = t_end - t_start;
    printf("%llu ns elapsed\n", ns);
    printf("%.6f GFlops achieved\n", ops / ns);
}

void obtain_platforms(cl_uint *const platforms_count,
                      cl_platform_id **const platform_ids) {
    cl_int result = clGetPlatformIDs(0, NULL, platforms_count);
    CHECK_ERR("obtain_platforms::clGetPlatformIDs", result, exit(0));

    printf("Platforms count: %d\n", *platforms_count);
    *platform_ids = (cl_platform_id *) malloc(*platforms_count * sizeof(cl_platform_id));

    result = clGetPlatformIDs(*platforms_count, *platform_ids, NULL);
    CHECK_ERR("obtain_platforms::clGetPlatformIDs", result, exit(0));

    //        Print info about platforms
    for (int i = 0; i < *platforms_count; ++i) {
        print_platform_info((*platform_ids)[i]);
    }
    printf("\n");
}


void find_suitable_devices(cl_uint platforms_count,
                           cl_platform_id *const platform_ids,
                           struct working_params *params) {
    for (int i = 0; i < platforms_count; ++i) {
        const cl_platform_id current_platform_id = platform_ids[i];

        cl_uint devices_count = 0;
        cl_int result = clGetDeviceIDs(current_platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &devices_count);
        CHECK_ERR("find_suitable_devices::clGetDeviceIDs", result, exit(0));

        printf("Platform %d. GPU devices count: %d\n", i, devices_count);
        cl_device_id *devices = (cl_device_id *) malloc(devices_count * sizeof(cl_device_id));

        result = clGetDeviceIDs(current_platform_id, CL_DEVICE_TYPE_GPU, devices_count, devices, NULL);
        CHECK_ERR("find_suitable_devices::clGetDeviceIDs", result, exit(0));

        filter_devices(&devices_count, devices);

        printf("Platform %d. Devices with not unified memory, count: %d\n", i, devices_count);



        /*for (int j = 0; j < devices_count; ++j) {
            print_device_info(devices[i]);
        }
        printf("\n");*/

        params->platform_id = current_platform_id;
        params->devices = devices;
        params->devices_count = max(1, devices_count);
        params->device = params->devices[0];

        if (devices_count > 0) {
            printf("Found suitable devices!\n");
            return;
        }
    }
    printf("Chosen integrated card!\n");
}

void create_context(struct working_params *params) {
    cl_int result;
    params->context = clCreateContext(NULL, params->devices_count, params->devices, NULL, NULL, &result);
    CHECK_ERR("create_context::clCreateContext", result, exit(0))

    if (params->context == NULL) {
        fprintf(stderr, "Created empty context!");
        exit(0);
    }
}

void create_command_queue(struct working_params *params) {
    cl_int result;

    params->command_queue_properties = CL_QUEUE_PROFILING_ENABLE;
    params->command_queue = clCreateCommandQueue(params->context, params->device, params->command_queue_properties,
                                                 &result);
    CHECK_ERR("create_command_queue::clCreateCommandQueue", result, exit(0))
    if (params->command_queue == NULL) {
        fprintf(stderr, "Created empty command queue!");
        exit(0);
    }
    printf("Created working command queue!\n");
}


void create_program(struct working_params *params, const char *kernel_file_name) {
    cl_int result;
    // ReSharper disable once CppDeprecatedEntity
    FILE *f = fopen(kernel_file_name, "rb");
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *code = (char *) (malloc(file_size * sizeof(char)));
    fread(code, 1, file_size, f);

    params->program = clCreateProgramWithSource(params->context, 1, (const char **) &code, &file_size, &result);
    CHECK_ERR("clCreateProgramWithSource", result, exit(0))

    fclose(f);
    free(code);

    printf("Program created!\n");
}

void build_program(struct working_params *params) {
    const cl_int result =
            clBuildProgram(params->program, params->devices_count, params->devices,
                           NULL, NULL, NULL);
    CHECK_ERR("build_program::clGetDeviceIDs", result, printf_build_error(params))
}


void create_kernel(struct working_params *params,
                   char *fun_name) {
    cl_int result;
    params->kernel = clCreateKernel(params->program, fun_name, &result);
    CHECK_ERR("create_kernel::clCreateKernel", result, exit(0))
}

cl_mem create_buffer(struct working_params *params,
                     size_t size,
                     cl_mem_flags flags) {
    cl_int result;
    const cl_mem mem = clCreateBuffer(params->context, flags, size, 0, &result);
    CHECK_ERR("create_buffer::clCreateBuffer", result, exit(0));
    return mem;
}

void enqueue_write_buffer(struct working_params *params,
                          cl_mem buffer,
                          size_t size,
                          const void *ptr) {
    const cl_int result = clEnqueueWriteBuffer(params->command_queue, buffer, CL_FALSE, 0, size, ptr, 0, NULL, NULL);
    CHECK_ERR("enqueue_write_buffer::clEnqueueWriteBuffer", result, exit(0));
}

void set_kernel_arg(struct working_params *params,
                    const int index,
                    size_t arg_size,
                    const void *arg_value) {
    const cl_int result = clSetKernelArg(params->kernel, index, arg_size, arg_value);
    CHECK_ERR("set_kernel_arg::clSetKernelArg", result, exit(0));
}

void enqueue_nd_range_kernel(struct working_params *params,
                             const size_t *const global_work_size,
                             const size_t *const local_work_size,
                             cl_event *run_event) {
    const cl_int result = clEnqueueNDRangeKernel(params->command_queue, params->kernel, 2, NULL,
                                                 global_work_size, local_work_size, 0, NULL, run_event);
    CHECK_ERR("enqueue_nd_range_kernel::clEnqueueNDRangeKernel", result, exit(0));
}

void enqueue_read_buffer(struct working_params *params,
                         cl_mem buffer,
                         size_t size,
                         void *ptr) {
    const cl_int result = clEnqueueReadBuffer(params->command_queue, buffer, CL_TRUE, 0, size, ptr, 0, NULL, NULL);
    CHECK_ERR("enqueue_read_buffer::clEnqueueReadBuffer", result, exit(0));
}

void run_algorithm(const char *file_name, float ops, cl_event (*algorithm)(struct working_params *)) {
    struct working_params params;

    cl_uint platforms_count;
    cl_platform_id *platform_ids = NULL;
    obtain_platforms(&platforms_count, &platform_ids);
    find_suitable_devices(platforms_count, platform_ids, &params);
    create_context(&params);
    create_command_queue(&params);
    char related_path[80] = "../";
    create_program(&params, strcat(related_path, file_name));
    build_program(&params);

    printf("Program was built!\n");

    cl_event run_event = algorithm(&params);

    print_profile_info(run_event, ops);
}