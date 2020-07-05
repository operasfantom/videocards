#include <stdbool.h>
#include <stdio.h>
#include <time.h>

#include <omp.h>

inline bool is_prime(const int x) {
    for (auto i = 2; i * i <= x; ++i) {
        if (x % i == 0) {
            return false;
        }
    }
    return true;
}

void count_primes() {
    static const int n = 10'000'000;

    // ReSharper disable once CppJoinDeclarationAndAssignment
    int counter = 1, i;
#pragma omp parallel for schedule(dynamic, 8) reduction(+:counter)
    // ReSharper disable once CppJoinDeclarationAndAssignment
    for (i = 3; i < n; ++i) {
        if (is_prime(i)) {
            ++counter;
        }
    }

    printf("Counter: %d\n", counter);
}

int main(int argc, char* argv[]) {
    const int max_threads = omp_get_max_threads();
    printf("Number of threads: %d\n", max_threads);

    clock_t start_t = clock();

    count_primes();

    clock_t end_t = clock();

    double total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("Total time: %.6f s", total_t);
    return 0;
}
