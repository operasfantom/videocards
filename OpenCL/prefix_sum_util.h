#ifndef OPENCL_PREFIX_SUM_UTIL_H
#define OPENCL_PREFIX_SUM_UTIL_H


#include "constants.h"

#include <stdio.h>
#include <stdlib.h>

#include <CL/opencl.h>

void generate_array(cl_int size,
                    size_t *memory_size,
                    cl_float **array,
                    cl_float **result_array);

void verify_prefix_sum(cl_int size,
                       const cl_float *array,
                       const cl_float *result);


#endif //OPENCL_PREFIX_SUM_UTIL_H
