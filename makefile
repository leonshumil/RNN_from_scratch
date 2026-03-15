CC = gcc
CFLAGS = -Wall -Iinclude -O3
LDFLAGS = -lm

SRC = $(wildcard src/*.c) main.c
OBJ = $(SRC:.c=.o)

all: rnn_project

rnn_project: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f src/*.o *.o rnn_project