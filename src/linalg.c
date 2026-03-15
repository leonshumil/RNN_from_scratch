#include "linalg.h"

void vec_add(float *a, const float *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

void mat_vec_mul(const float *W, const float *x, const float *b, float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        // Start with the bias value
        float sum = (b != NULL) ? b[i] : 0.0f;
        
        // Dot product of W[row i] and vector x
        for (int j = 0; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }
        out[i] = sum;
    }
}

void random_fill(float *arr, int size, float r) {
    for (int i = 0; i < size; i++) {
        // Generate random float between -r and r
        arr[i] = ((float)rand() / (float)RAND_MAX) * 2.0f * r - r;
    }
}

void vec_elementwise_mul(float *a, const float *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] *= b[i];
    }
}