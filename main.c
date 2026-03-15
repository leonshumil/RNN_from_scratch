#include <stdio.h>
#include <time.h>
#include "rnn.h"

int main() {
    
    int rows, cols;
    float *train_data = load_csv("data/silver.csv", &rows, &cols);

    if (train_data) {
    printf("Loaded %d rows and %d columns from silver.csv\n", rows, cols);
    // Proceed to forward pass with train_data...
}
    // 1. Setup Seed for reproducibility (using time for now)
    srand(time(NULL));

    // 2. Hyperparameters
    int input_dim = 3;    // e.g., 3 features per time step
    int hidden_dim = 5;   // 5 "memory" neurons
    int output_dim = 2;   // 2 possible output categories
    int seq_len = 4;      // A short sequence of 4 time steps

    // 3. Memory Allocation for sequence data
    // Input: seq_len * input_dim
    float *input_seq = malloc(seq_len * input_dim * sizeof(float));
    // Hidden History: seq_len * hidden_dim (CRITICAL for BPTT later)
    float *h_history = malloc(seq_len * hidden_dim * sizeof(float));
    // Output: seq_len * output_dim
    float *y_out = malloc(seq_len * output_dim * sizeof(float));

    // 4. Fill dummy input with random values
    printf("--- Input Sequence ---\n");
    for (int t = 0; t < seq_len; t++) {
        printf("t=%d: [ ", t);
        for (int i = 0; i < input_dim; i++) {
            input_seq[t * input_dim + i] = (float)rand() / RAND_MAX;
            printf("%.2f ", input_seq[t * input_dim + i]);
        }
        printf("]\n");
    }

    // 5. Initialize Model
    RNN *my_rnn = init_rnn(input_dim, hidden_dim, output_dim);
    printf("\nRNN Structure: %d (In) -> %d (Hidden) -> %d (Out)\n", 
           input_dim, hidden_dim, output_dim);

    // 6. Run Forward Pass
    printf("\nRunning Forward Pass...\n");
    rnn_forward(my_rnn, input_seq, seq_len, h_history, y_out);

    // 7. Inspect Output
    printf("\n--- Model Output (y_t) ---\n");
    for (int t = 0; t < seq_len; t++) {
        printf("t=%d: [ ", t);
        for (int i = 0; i < output_dim; i++) {
            printf("%.4f ", y_out[t * output_dim + i]);
        }
        printf("]\n");
    }

    // 8. Cleanup
    free(input_seq);
    free(h_history);
    free(y_out);
    free_rnn(my_rnn);

    printf("\nSuccess! The forward pass executed without crashing.\n");

    return 0;
}