#include "utils.h"
#include <math.h>

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_deriv(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

float tanh_act(float x) {
    return tanhf(x);
}

float tanh_deriv(float x) {
    float t = tanhf(x);
    return 1.0f - t * t;
}

void softmax(float *arr, int size) {
    float max_val = arr[0];
    for (int i = 1; i < size; i++) if (arr[i] > max_val) max_val = arr[i];

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        arr[i] = expf(arr[i] - max_val); // Numerical stability trick
        sum += arr[i];
    }
    for (int i = 0; i < size; i++) {
        arr[i] /= sum;
    }
}