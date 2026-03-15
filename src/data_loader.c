#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float* load_csv(const char* filename, int *rows, int *cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        return NULL;
    }

    char line[1024];
    int r = 0, c = 0;

    // First pass: count rows and columns
    if (fgets(line, sizeof(line), file)) {
        char *tmp = strdup(line);
        char *token = strtok(tmp, ",");
        while (token) {
            c++;
            token = strtok(NULL, ",");
        }
        free(tmp);
    }
    rewind(file);
    while (fgets(line, sizeof(line), file)) r++;
    rewind(file);

    *rows = r;
    *cols = c;
    float *data = malloc(r * c * sizeof(float));

    // Second pass: load data
    int i = 0;
    while (fgets(line, sizeof(line), file)) {
        char *token = strtok(line, ",");
        while (token) {
            data[i++] = atof(token);
            token = strtok(NULL, ",");
        }
    }

    fclose(file);
    return data;
}
void normalize_data(float *data, int rows, int cols, int target_col, float *out_min, float *out_max) {
    float min = data[target_col];
    float max = data[target_col];

    // Find min/max for the specific column we are predicting
    for (int i = 0; i < rows; i++) {
        float val = data[i * cols + target_col];
        if (val < min) min = val;
        if (val > max) max = val;
    }

    // Perform the normalization across the whole dataset (or just that column)
    for (int i = 0; i < rows * cols; i++) {
        // Simple global scaling for this example
        data[i] = (data[i] - min) / (max - min + 1e-7);
    }

    *out_min = min;
    *out_max = max;
}