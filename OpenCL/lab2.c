#define CL_TARGET_OPENCL_VERSION 220

#include "constants.h"
#include "cl_util.h"
#include "prefix_sum.h"

int main() {
    run_algorithm("prefix_sum.cl", 2.0f * L, buffered_prefix_sum);
    return 0;
}

