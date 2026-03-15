#include <stdio.h>
#include <assert.h>
#include "linalg.h"

void test_matrix_mul() {
    float W[4] = {1, 2, 3, 4}; // 2x2
    float v[2] = {1, 1};       // 2x1
    float b[2] = {0, 0};
    float out[2];

    mat_vec_mul(W, v, b, out, 2, 2);

    // Expected: [1*1 + 2*1, 3*1 + 4*1] = [3, 7]
    assert(out[0] == 3.0f);
    assert(out[1] == 7.0f);
    printf("Matrix Multiplication Test: PASSED\n");
}

int main() {
    test_matrix_mul();
    // Add tests for tanh_act, normalize_data, etc.
    return 0;
}