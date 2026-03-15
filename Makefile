CC = gcc
CFLAGS = -Wall -Wextra -Iinclude -O3
LDFLAGS = -lm

TARGET = rnn_project
TEST_TARGET = rnn_test
SRC_DIR = src
OBJ_DIR = obj

SOURCES = $(wildcard $(SRC_DIR)/*.c)
# Exclude main.c from general objects so we can link it specifically
COMMON_OBJS = $(SOURCES:%.c=$(OBJ_DIR)/%.o)

all: $(TARGET)

$(TARGET): $(COMMON_OBJS) $(OBJ_DIR)/main.o
	$(CC) -o $@ $^ $(LDFLAGS)

# Rule for building the test runner
test: $(COMMON_OBJS) $(OBJ_DIR)/test/test_model.o
	$(CC) -o $(TEST_TARGET) $^ $(LDFLAGS)
	./$(TEST_TARGET)

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET) $(TEST_TARGET)

.PHONY: all clean test