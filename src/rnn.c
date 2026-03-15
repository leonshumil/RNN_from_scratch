#include "rnn.h"   
#include "linalg.h"
#include "utils.h"
#include <stdio.h>    
#include <stdlib.h>
#include <string.h>
#include <math.h>  

// --- NEW: Save function to resolve linker error ---
void save_weights(RNN *net, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror("Failed to open file for saving");
        return;
    }
    fwrite(net->Wxh, sizeof(float), net->hidden_dim * net->input_dim, f);
    fwrite(net->Whh, sizeof(float), net->hidden_dim * net->hidden_dim, f);
    fwrite(net->Why, sizeof(float), net->output_dim * net->hidden_dim, f);
    fwrite(net->bh, sizeof(float), net->hidden_dim, f);
    fwrite(net->by, sizeof(float), net->output_dim, f);
    fclose(f);
    printf("Weights and biases saved to %s\n", filename);
}

void load_weights(RNN *net, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open file for loading");
        return;
    }
    // Added explicit return value checks to satisfy -Wunused-result
    if (fread(net->Wxh, sizeof(float), net->hidden_dim * net->input_dim, f) == 0) perror("fread Wxh");
    if (fread(net->Whh, sizeof(float), net->hidden_dim * net->hidden_dim, f) == 0) perror("fread Whh");
    if (fread(net->Why, sizeof(float), net->output_dim * net->hidden_dim, f) == 0) perror("fread Why");
    if (fread(net->bh, sizeof(float), net->hidden_dim, f) == 0) perror("fread bh");
    if (fread(net->by, sizeof(float), net->output_dim, f) == 0) perror("fread by");
    
    fclose(f);
    printf("Weights and biases loaded from %s\n", filename);
}

RNN* init_rnn(int in, int hid, int out) {
    RNN *net = malloc(sizeof(RNN));
    net->input_dim = in;
    net->hidden_dim = hid;
    net->output_dim = out;

    net->Wxh = malloc(hid * in * sizeof(float));
    net->Whh = malloc(hid * hid * sizeof(float));
    net->Why = malloc(out * hid * sizeof(float));
    net->bh = calloc(hid, sizeof(float));
    net->by = calloc(out, sizeof(float));

    random_fill(net->Wxh, hid * in, sqrtf(1.0f / in));
    random_fill(net->Whh, hid * hid, sqrtf(1.0f / hid));
    random_fill(net->Why, out * hid, sqrtf(1.0f / hid));

    return net;
}

void rnn_forward(RNN *net, float *input_seq, int seq_len, float *h_history, float *y_out) {
    int h_dim = net->hidden_dim;
    int in_dim = net->input_dim;
    int out_dim = net->output_dim;

    float *prev_h = calloc(h_dim, sizeof(float));
    float *current_h_ptr;

    for (int t = 0; t < seq_len; t++) {
        current_h_ptr = &h_history[t * h_dim];
        
        float *xt = &input_seq[t * in_dim];
        mat_vec_mul(net->Wxh, xt, net->bh, current_h_ptr, h_dim, in_dim);

        float *temp_hh = malloc(h_dim * sizeof(float));
        mat_vec_mul(net->Whh, prev_h, NULL, temp_hh, h_dim, h_dim);
        
        for (int i = 0; i < h_dim; i++) {
            current_h_ptr[i] = tanh_act(current_h_ptr[i] + temp_hh[i]);
        }

        mat_vec_mul(net->Why, current_h_ptr, net->by, &y_out[t * out_dim], out_dim, h_dim);

        memcpy(prev_h, current_h_ptr, h_dim * sizeof(float));
        free(temp_hh);
    }
    free(prev_h);
}

void rnn_backward(RNN *net, float *input_seq, float *h_history, float *y_errors, int seq_len, float lr) {
    int h_dim = net->hidden_dim;
    int in_dim = net->input_dim;
    int out_dim = net->output_dim;

    float *dWxh = calloc(h_dim * in_dim, sizeof(float));
    float *dWhh = calloc(h_dim * h_dim, sizeof(float));
    float *dWhy = calloc(out_dim * h_dim, sizeof(float));
    float *dbh  = calloc(h_dim, sizeof(float));
    float *dby  = calloc(out_dim, sizeof(float));
    float *dh_next = calloc(h_dim, sizeof(float));

    for (int t = seq_len - 1; t >= 0; t--) {
        float *dy = &y_errors[t * out_dim];
        float *ht = &h_history[t * h_dim];

        for (int i = 0; i < out_dim; i++) {
            for (int j = 0; j < h_dim; j++) {
                dWhy[i * h_dim + j] += dy[i] * ht[j];
            }
            dby[i] += dy[i];
        }

        float *dh = malloc(h_dim * sizeof(float));
        for (int i = 0; i < h_dim; i++) {
            dh[i] = dh_next[i];
            for (int j = 0; j < out_dim; j++) {
                dh[i] += net->Why[j * h_dim + i] * dy[j];
            }
        }

        float *dh_raw = malloc(h_dim * sizeof(float));
        for (int i = 0; i < h_dim; i++) {
            dh_raw[i] = dh[i] * (1.0f - ht[i] * ht[i]);
        }

        float *xt = &input_seq[t * in_dim];
        float *h_prev = (t > 0) ? &h_history[(t - 1) * h_dim] : calloc(h_dim, sizeof(float));

        for (int i = 0; i < h_dim; i++) {
            for (int j = 0; j < in_dim; j++) dWxh[i * in_dim + j] += dh_raw[i] * xt[j];
            for (int j = 0; j < h_dim; j++) dWhh[i * h_dim + j] += dh_raw[i] * h_prev[j];
            dbh[i] += dh_raw[i];
        }

        for (int i = 0; i < h_dim; i++) {
            dh_next[i] = 0;
            for (int j = 0; j < h_dim; j++) {
                dh_next[i] += net->Whh[j * h_dim + i] * dh_raw[j];
            }
        }
        
        free(dh); free(dh_raw);
        if (t == 0) free(h_prev);
    }

    for (int i = 0; i < h_dim * in_dim; i++) net->Wxh[i] -= lr * dWxh[i];
    for (int i = 0; i < h_dim * h_dim; i++) net->Whh[i] -= lr * dWhh[i];
    for (int i = 0; i < out_dim * h_dim; i++) net->Why[i] -= lr * dWhy[i];
    for (int i = 0; i < h_dim; i++) net->bh[i] -= lr * dbh[i];
    for (int i = 0; i < out_dim; i++) net->by[i] -= lr * dby[i];

    free(dWxh); free(dWhh); free(dWhy); free(dbh); free(dby); free(dh_next);
}

void free_rnn(RNN *net) {
    free(net->Wxh); free(net->Whh); free(net->Why);
    free(net->bh); free(net->by);
    free(net);
}