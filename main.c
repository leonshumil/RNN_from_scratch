#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "rnn.h"
#include "utils.h"

#define WINDOW_SIZE 10
#define FEATURE_COL 4    // Based on your CSV, 'Close' is the 5th column (index 4)
#define EPOCHS 100
#define LEARNING_RATE 0.01f

float* load_csv(const char* filename, int *rows, int *cols);
void normalize_data(float *data, int rows, int cols, int target_col, float *out_min, float *out_max);

float calculate_mse(float pred, float target) {
    return 0.5f * powf(pred - target, 2);
}

int main() {
    srand(time(NULL));

    int rows, cols;
    float min_val, max_val;
    
    printf("Loading silver.csv...\n");
    float *data = load_csv("data/silver.csv", &rows, &cols);
    if (!data) return 1;

    // Pass addresses of min_val and max_val to capture them
    normalize_data(data, rows, cols, FEATURE_COL, &min_val, &max_val);

    int input_dim = 1;
    int hidden_dim = 16;
    int output_dim = 1;
    RNN *net = init_rnn(input_dim, hidden_dim, output_dim);

    printf("Starting training on %d rows...\n", rows);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0;
        int count = 0;

        for (int i = 0; i < rows - WINDOW_SIZE - 1; i++) {
            float *h_history = calloc(WINDOW_SIZE * hidden_dim, sizeof(float));
            float *y_out = calloc(WINDOW_SIZE * output_dim, sizeof(float));
            float *y_errors = calloc(WINDOW_SIZE * output_dim, sizeof(float));
            float *x_window = malloc(WINDOW_SIZE * input_dim * sizeof(float));

            for (int t = 0; t < WINDOW_SIZE; t++) {
                x_window[t] = data[(i + t) * cols + FEATURE_COL];
            }

            float target = data[(i + WINDOW_SIZE) * cols + FEATURE_COL];

            rnn_forward(net, x_window, WINDOW_SIZE, h_history, y_out);

            float final_pred = y_out[(WINDOW_SIZE - 1) * output_dim];
            y_errors[(WINDOW_SIZE - 1) * output_dim] = final_pred - target;
            epoch_loss += calculate_mse(final_pred, target);

            rnn_backward(net, x_window, h_history, y_errors, WINDOW_SIZE, LEARNING_RATE);

            free(x_window); free(h_history); free(y_out); free(y_errors);
            count++;
        }

        if (epoch % 10 == 0) printf("Epoch %d: Average Loss = %f\n", epoch, epoch_loss / count);
    }

    printf("Training Complete.\n");

    // --- PREDICTION FOR TOMORROW ---
    printf("\n--- Final Prediction ---\n");

    float *last_window = malloc(WINDOW_SIZE * input_dim * sizeof(float));
    // The very last WINDOW_SIZE days in the file
    for (int t = 0; t < WINDOW_SIZE; t++) {
        int row_idx = (rows - WINDOW_SIZE) + t;
        last_window[t] = data[row_idx * cols + FEATURE_COL];
    }

    float *final_h_hist = calloc(WINDOW_SIZE * hidden_dim, sizeof(float));
    float *final_y_out = calloc(WINDOW_SIZE * output_dim, sizeof(float));

    rnn_forward(net, last_window, WINDOW_SIZE, final_h_hist, final_y_out);

    float norm_pred = final_y_out[(WINDOW_SIZE - 1) * output_dim];

    // Convert back to US Dollars
    float actual_prediction = (norm_pred * (max_val - min_val)) + min_val;

    printf("Last known price: $%.2f\n", (data[(rows-1)*cols + FEATURE_COL] * (max_val - min_val)) + min_val);
    printf("Predicted Silver Price for Tomorrow: $%.2f\n", actual_prediction);

    free(last_window); free(final_h_hist); free(final_y_out);
    free_rnn(net); free(data);

    return 0;
}