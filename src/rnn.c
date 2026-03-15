#include "rnn.h"
#include <string.h>

RNN* init_rnn(int in, int hid, int out) {
    RNN *net = malloc(sizeof(RNN));
    net->input_dim = in;
    net->hidden_dim = hid;
    net->output_dim = out;

    // Allocate
    net->Wxh = malloc(hid * in * sizeof(float));
    net->Whh = malloc(hid * hid * sizeof(float));
    net->Why = malloc(out * hid * sizeof(float));
    net->bh = calloc(hid, sizeof(float));
    net->by = calloc(out, sizeof(float));

    // Initialise with Xavier/He scaling
    random_fill(net->Wxh, hid * in, sqrtf(1.0f / in));
    random_fill(net->Whh, hid * hid, sqrtf(1.0f / hid));
    random_fill(net->Why, out * hid, sqrtf(1.0f / hid));

    return net;
}

void rnn_forward(RNN *net, float *input_seq, int seq_len, float *h_history, float *y_out) {
    int h_dim = net->hidden_dim;
    int in_dim = net->input_dim;
    int out_dim = net->output_dim;

    // We start with a zeroed-out hidden state for t = -1
    float *prev_h = calloc(h_dim, sizeof(float));
    float *current_h_ptr;

    for (int t = 0; t < seq_len; t++) {
        // Point to the correct slot in history for this time step
        current_h_ptr = &h_history[t * h_dim];
        
        // 1. Calculate Wxh * x_t + bh
        float *xt = &input_seq[t * in_dim];
        mat_vec_mul(net->Wxh, xt, net->bh, current_h_ptr, h_dim, in_dim);

        // 2. Add Whh * h_{t-1}
        float *temp_hh = malloc(h_dim * sizeof(float));
        mat_vec_mul(net->Whh, prev_h, NULL, temp_hh, h_dim, h_dim);
        
        // 3. Apply activation: h_t = tanh(Wxh*xt + Whh*ht-1 + bh)
        for (int i = 0; i < h_dim; i++) {
            current_h_ptr[i] = tanh_act(current_h_ptr[i] + temp_hh[i]);
        }

        // 4. Calculate output y_t = Why * h_t + by
        mat_vec_mul(net->Why, current_h_ptr, net->by, &y_out[t * out_dim], out_dim, h_dim);

        // Update prev_h for next iteration
        memcpy(prev_h, current_h_ptr, h_dim * sizeof(float));
        free(temp_hh);
    }
    free(prev_h);
}

void free_rnn(RNN *net) {
    free(net->Wxh); free(net->Whh); free(net->Why);
    free(net->bh); free(net->by);
    free(net);
}