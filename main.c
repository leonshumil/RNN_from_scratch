#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "rnn.h"
#include "utils.h"

// Hyperparameters
#define WINDOW_SIZE 10
#define INPUT_DIM 5      // Open, High, Low, Close, Volume
#define OUTPUT_DIM 1     // Target: Next Close
#define TARGET_COL 4     // Index of 'Close' in the normalized data
#define EPOCHS 100
#define LEARNING_RATE 0.005f

// External function declarations from your other .c files
float* load_csv(const char* filename, int *rows, int *cols);
void normalize_all_data(float *data, int rows, int cols, int target_col, float *out_min, float *out_max);

float calculate_mse(float pred, float target) {
    return 0.5f * powf(pred - target, 2);
}

int main() {
    srand(time(NULL));

    int rows, cols;
    float min_val, max_val; // Stores target column scaling factors
    
    printf("--- RNN Silver Price Predictor ---\n");
    printf("Loading data/silver.csv...\n");
    float *data = load_csv("data/silver.csv", &rows, &cols);
    if (!data) return 1;

    // Normalizing all 5 features independently to [0, 1]
    normalize_all_data(data, rows, cols, TARGET_COL, &min_val, &max_val);

    int hidden_dim = 32; 
    RNN *net = init_rnn(INPUT_DIM, hidden_dim, OUTPUT_DIM);

    // Split logic (80% Train, 20% Test)
    int train_rows = (int)(rows * 0.8);
    printf("Data rows: %d | Features: %d\n", rows, INPUT_DIM);
    printf("Training on %d rows, Testing on remaining %d rows.\n\n", train_rows, rows - train_rows);

    // --- TRAINING LOOP ---
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float train_loss = 0;
        int count = 0;

        for (int i = 0; i < train_rows - WINDOW_SIZE - 1; i++) {
            float *h_history = calloc(WINDOW_SIZE * hidden_dim, sizeof(float));
            float *y_out = calloc(WINDOW_SIZE * OUTPUT_DIM, sizeof(float));
            float *y_errors = calloc(WINDOW_SIZE * OUTPUT_DIM, sizeof(float));
            float *x_window = malloc(WINDOW_SIZE * INPUT_DIM * sizeof(float));

            // Populate Multi-variate Window (Columns 1-5)
            for (int t = 0; t < WINDOW_SIZE; t++) {
                for (int f = 0; f < INPUT_DIM; f++) {
                    x_window[t * INPUT_DIM + f] = data[(i + t) * cols + (1 + f)];
                }
            }

            float target = data[(i + WINDOW_SIZE) * cols + TARGET_COL];
            rnn_forward(net, x_window, WINDOW_SIZE, h_history, y_out);

            float final_pred = y_out[(WINDOW_SIZE - 1) * OUTPUT_DIM];
            y_errors[(WINDOW_SIZE - 1) * OUTPUT_DIM] = final_pred - target;
            train_loss += calculate_mse(final_pred, target);

            rnn_backward(net, x_window, h_history, y_errors, WINDOW_SIZE, LEARNING_RATE);

            free(x_window); free(h_history); free(y_out); free(y_errors);
            count++;
        }

        if (epoch % 10 == 0 || epoch == EPOCHS - 1) {
            printf("Epoch %d: Average Training Loss = %f\n", epoch, train_loss / count);
        }
    }

    printf("\n--- TESTING ON UNSEEN DATA ---\n");
    float total_mae = 0;
    int correct_direction = 0;
    int test_samples = 0;

    for (int i = train_rows; i < rows - WINDOW_SIZE - 1; i++) {
        float *h_history = calloc(WINDOW_SIZE * hidden_dim, sizeof(float));
        float *y_out = calloc(WINDOW_SIZE * OUTPUT_DIM, sizeof(float));
        float *x_window = malloc(WINDOW_SIZE * INPUT_DIM * sizeof(float));

        for (int t = 0; t < WINDOW_SIZE; t++) {
            for (int f = 0; f < INPUT_DIM; f++) {
                x_window[t * INPUT_DIM + f] = data[(i + t) * cols + (1 + f)];
            }
        }

        float target_norm = data[(i + WINDOW_SIZE) * cols + TARGET_COL];
        float prev_close_norm = data[(i + WINDOW_SIZE - 1) * cols + TARGET_COL];

        rnn_forward(net, x_window, WINDOW_SIZE, h_history, y_out);
        float pred_norm = y_out[(WINDOW_SIZE - 1) * OUTPUT_DIM];

        // 1. MAE in USD
        float pred_usd = (pred_norm * (max_val - min_val)) + min_val;
        float target_usd = (target_norm * (max_val - min_val)) + min_val;
        total_mae += fabsf(pred_usd - target_usd);

        // 2. Directional Accuracy (Did we guess the movement?)
        int actual_up = (target_norm > prev_close_norm);
        int pred_up = (pred_norm > prev_close_norm);
        if (actual_up == pred_up) correct_direction++;

        free(x_window); free(h_history); free(y_out);
        test_samples++;
    }

    printf("Test Results:\n");
    printf(">> Mean Absolute Error: $%.2f\n", total_mae / test_samples);
    printf(">> Directional Accuracy: %.2f%%\n", ((float)correct_direction / test_samples) * 100.0f);

    // --- TOMORROW'S FORECAST ---
    printf("\n--- Forecast for Tomorrow ---\n");
    float *tomorrow_window = malloc(WINDOW_SIZE * INPUT_DIM * sizeof(float));
    for (int t = 0; t < WINDOW_SIZE; t++) {
        for (int f = 0; f < INPUT_DIM; f++) {
            tomorrow_window[t * INPUT_DIM + f] = data[(rows - WINDOW_SIZE + t) * cols + (1 + f)];
        }
    }

    float *final_h = calloc(WINDOW_SIZE * hidden_dim, sizeof(float));
    float *final_y = calloc(WINDOW_SIZE * OUTPUT_DIM, sizeof(float));
    rnn_forward(net, tomorrow_window, WINDOW_SIZE, final_h, final_y);

    float pred_final = (final_y[(WINDOW_SIZE - 1) * OUTPUT_DIM] * (max_val - min_val)) + min_val;
    float last_actual = (data[(rows - 1) * cols + TARGET_COL] * (max_val - min_val)) + min_val;

    printf("Last Price: $%.2f | Predicted Tomorrow: $%.2f\n", last_actual, pred_final);
    printf("Signal: %s\n", (pred_final > last_actual) ? "BUY (UP)" : "SELL (DOWN)");

    // Save for future use
    save_weights(net, "trained_silver_rnn.bin");

    // Cleanup
    free(tomorrow_window); free(final_h); free(final_y);
    free_rnn(net);
    free(data);

    return 0;
}