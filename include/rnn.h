#ifndef RNN_H
#define RNN_H

typedef struct {
    int input_dim;
    int hidden_dim;
    int output_dim;
    float *Wxh, *Whh, *Why;
    float *bh, *by;
} RNN;

RNN* init_rnn(int in, int hid, int out);
// Processes a full sequence and stores all hidden states for later BPTT
void rnn_forward(RNN *net, float *input_seq, int seq_len, float *h_history, float *y_out);
void rnn_backward(RNN *net, float *input_seq, float *h_history, float *y_errors, int seq_len, float lr);
void free_rnn(RNN *net);

#endif