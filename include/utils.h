#ifndef UTILS_H
#define UTILS_H

#include <math.h>

// Activation Functions
float sigmoid(float x);
float tanh_act(float x); // Named tanh_act to avoid conflict with math.h tanh

// Derivatives (for Backprop)
float sigmoid_deriv(float x);
float tanh_deriv(float x);

// Specialized for output layers
void softmax(float *arr, int size);

#endif