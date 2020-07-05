#ifndef OPENCL_MATRIX_UTIL_H
#define OPENCL_MATRIX_UTIL_H

#include <CL/cl_platform.h>

void generate_matrix(cl_int first_size,
                     cl_int second_size,
                     cl_int third_size,
                     size_t *first_memory_size,
                     size_t *second_memory_size,
                     size_t *result_memory_size,
                     cl_float **first_matrix,
                     cl_float **second_matrix,
                     cl_float **result_matrix);

void verify_matrix_multiplication(cl_int first_size,
                                  cl_int second_size,
                                  cl_int third_size,
                                  const cl_float *first_matrix,
                                  const cl_float *second_matrix,
                                  const cl_float *result_matrix);


#endif //OPENCL_MATRIX_UTIL_H
