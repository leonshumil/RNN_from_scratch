# Compiler and Flags
CC = gcc
CFLAGS = -Wall -Wextra -Iinclude -O3
LDFLAGS = -lm

# Project Structure
TARGET = rnn_project
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj

# Find all source files and convert to object file paths
SOURCES = $(wildcard $(SRC_DIR)/*.c) main.c
OBJECTS = $(SOURCES:%.c=$(OBJ_DIR)/%.o)

# Default rule
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Rule to compile .c files into .o files
$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean