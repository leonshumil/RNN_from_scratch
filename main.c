#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "rnn.h"
#include "utils.h"

// Define these based on your silver.csv structure
#define WINDOW_SIZE 10    // Number of past days to look at
#define FEATURE_COL 3     // Assuming 'Close' price is column 3
#define EPOCHS 100
#define LEARNING_RATE 0.01f

// Helper to calculate Mean Squared Error loss
float calculate_mse(float pred, float target) {
    return 0.5f * powf(pred - target, 2);
}

int main() {
    srand(time(NULL));

    // 1. Load and Preprocess Data
    int rows, cols;
    printf("Loading silver.csv...\n");
    float *data = load_csv("data/silver.csv", &rows, &cols);
    if (!data) return 1;

    // Normalizing is required for tanh stability
    normalize_data(data, rows, cols);

    // 2. Initialize RNN
    // input_dim = 1 (we are just looking at the price)
    // output_dim = 1 (predicting the next price)
    int input_dim = 1;
    int hidden_dim = 16;
    int output_dim = 1;
    RNN *net = init_rnn(input_dim, hidden_dim, output_dim);

    printf("Starting training on %d rows...\n", rows);

    // 3. Training Loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0;
        int count = 0;

        // Iterate through data using a sliding window
        for (int i = 0; i < rows - WINDOW_SIZE - 1; i++) {
            // Memory for this specific sequence
            float *h_history = calloc(WINDOW_SIZE * hidden_dim, sizeof(float));
            float *y_out = calloc(WINDOW_SIZE * output_dim, sizeof(float));
            float *y_errors = calloc(WINDOW_SIZE * output_dim, sizeof(float));
            
            // Extract the window of input features
            float *x_window = malloc(WINDOW_SIZE * input_dim * sizeof(float));
            for (int t = 0; t < WINDOW_SIZE; t++) {
                x_window[t] = data[(i + t) * cols + FEATURE_COL];
            }

            // Target is the price of the DAY AFTER the window ends
            float target = data[(i + WINDOW_SIZE) * cols + FEATURE_COL];

            // --- FORWARD PASS ---
            rnn_forward(net, x_window, WINDOW_SIZE, h_history, y_out);

            // Calculate error only for the last prediction in the sequence (Many-to-One)
            float final_pred = y_out[(WINDOW_SIZE - 1) * output_dim];
            y_errors[(WINDOW_SIZE - 1) * output_dim] = final_pred - target;
            
            epoch_loss += calculate_mse(final_pred, target);

            // --- BACKWARD PASS (BPTT) ---
            rnn_backward(net, x_window, h_history, y_errors, WINDOW_SIZE, LEARNING_RATE);

            // Cleanup step memory
            free(x_window);
            free(h_history);
            free(y_out);
            free(y_errors);
            count++;
        }

        if (epoch % 10 == 0) {
            printf("Epoch %d: Average Loss = %f\n", epoch, epoch_loss / count);
        }
    }

    printf("Training Complete.\n");

    // 4. Cleanup
    free_rnn(net);
    free(data);

    return 0;
}