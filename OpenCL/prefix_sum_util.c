#include "prefix_sum_util.h"

#include "math.h"

float generate_cell() {
    return rand() / RAND_MAX;
}

void generate_array(cl_int size, size_t *memory_size, cl_float **array, cl_float **result_array) {
    *memory_size = size * sizeof(cl_float);
    *array = (cl_float *) malloc(*memory_size);
    *result_array = (cl_float *) malloc(*memory_size);

    for (int i = 0; i < size; ++i) {
        (*array)[i] = generate_cell();
        (*result_array)[i] = 0;
    }
}

void verify_prefix_sum(cl_int size, const cl_float *array, const cl_float *result) {
    printf("\n\nVerifying prefix sum started\n");

    cl_float sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += array[i];
        cl_float expected = sum;
        cl_float actual = result[i];
        if (fabsf(expected - actual) > precision) {
            fprintf(stderr, "Expected: %.5f, actual: %.5f. Element in position %d\n", expected, actual, i);
        }
    }

    printf("Verifying prefix sum completed\n\n");
}
