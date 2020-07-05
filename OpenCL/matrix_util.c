#include "matrix_util.h"

#include "constants.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float generate_cell() {
    return rand() / RAND_MAX;
}

void generate_matrix(cl_int first_size, cl_int second_size, cl_int third_size, size_t *first_memory_size,
                     size_t *second_memory_size, size_t *result_memory_size, cl_float **first_matrix,
                     cl_float **second_matrix, cl_float **result_matrix) {
    srand(0);
    
    *first_memory_size = first_size * second_size * sizeof(cl_float);
    *second_memory_size = second_size * third_size * sizeof(cl_float);
    *result_memory_size = first_size * third_size * sizeof(cl_float);

    *first_matrix = (cl_float *) malloc(*first_memory_size);
    *second_matrix = (cl_float *) malloc(*second_memory_size);
    *result_matrix = (cl_float *) malloc(*result_memory_size);

    for (int i = 0; i < first_size; ++i) {
        for (int j = 0; j < second_size; ++j) {
            cl_float *cell = &(*first_matrix)[i * second_size + j];
            *cell = generate_cell();
        }
    }
    for (int i = 0; i < second_size; ++i) {
        for (int j = 0; j < third_size; ++j) {
            cl_float *cell = &(*second_matrix)[i * third_size + j];
            *cell = generate_cell();
        }
    }
}

void verify_matrix_multiplication(cl_int first_size,
                                  cl_int second_size,
                                  cl_int third_size,
                                  const cl_float *first_matrix,
                                  const cl_float *second_matrix,
                                  const cl_float *result_matrix) {
    printf("\n\nVerifying matrix started\n");
    for (int i = 0; i < first_size; ++i) {
        for (int j = 0; j < third_size; ++j) {
            cl_float expected = 0;
            for (int k = 0; k < second_size; ++k) {
                expected += first_matrix[i * second_size + k] * second_matrix[k * third_size + j];
            }
            const cl_float actual = result_matrix[i * third_size + j];
            if (fabsf(expected - actual) > precision) {
                fprintf(stderr, "Expected: %.5f, actual: %.5f. Element in row %d, column %d\n", expected, actual, i, j);
            }
        }
    }
    printf("Verifying matrix completed\n\n");
}
