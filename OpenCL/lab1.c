#define CL_TARGET_OPENCL_VERSION 220

#include "matrix_multiplication.h"
#include "constants.h"

int main() {
    run_algorithm("matrix_multiplication.cl", 2.0f * N * M * K, buffered_matrix_multiplication);
    return 0;
}

