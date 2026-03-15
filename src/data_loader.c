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
void normalize_data(float *data, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        float min = data[j], max = data[j];
        for (int i = 0; i < rows; i++) {
            if (data[i * cols + j] < min) min = data[i * cols + j];
            if (data[i * cols + j] > max) max = data[i * cols + j];
        }
        for (int i = 0; i < rows; i++) {
            data[i * cols + j] = (data[i * cols + j] - min) / (max - min + 1e-7);
        }
    }
}