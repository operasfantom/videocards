#ifndef OPENCL_CONSTANTS_H
#define OPENCL_CONSTANTS_H

#include <stdio.h>

#define max(a, b) ((a) > (b) ? (a) : (b))

extern const float precision;

extern size_t TILE_W;
extern size_t TILE_H;
extern size_t WORK_PER_THREAD;

extern const int N;
extern const int M;
extern const int K;

extern const int L;

#endif //OPENCL_CONSTANTS_H
