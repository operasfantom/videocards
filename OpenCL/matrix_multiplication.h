#ifndef OPENCL_MATRIX_MULTIPLICATION_H
#define OPENCL_MATRIX_MULTIPLICATION_H

#include "cl_util.h"

#include <CL/opencl.h>

cl_event buffered_matrix_multiplication(struct working_params *params);

#endif //OPENCL_MATRIX_MULTIPLICATION_H
