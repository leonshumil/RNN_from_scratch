#ifndef LINALG_H
#define LINALG_H

#include <stdlib.h>
#include <math.h>

// Vector Addition: a = a + b
void vec_add(float *a, const float *b, int size);

// Matrix-Vector Multiplication: out = (W * x) + b
// W: matrix (rows x cols)
// x: input vector (cols)
// b: bias vector (rows)
void mat_vec_mul(const float *W, const float *x, const float *b, float *out, int rows, int cols);

// Element-wise multiplication (for Hadamard products in GRU/LSTM)
void vec_elementwise_mul(float *a, const float *b, int size);

// Fill with random values in range [-r, r]
void random_fill(float *arr, int size, float r);

#endif