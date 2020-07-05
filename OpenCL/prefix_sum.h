#ifndef OPENCL_PREFIX_SUM_H
#define OPENCL_PREFIX_SUM_H

#include "constants.h"
#include "prefix_sum_util.h"
#include "cl_util.h"

#include <CL/opencl.h>

cl_event buffered_prefix_sum(struct working_params *params);

#endif //OPENCL_PREFIX_SUM_H
