# Compiler to use
CC := gcc

# Compiler flags
CFLAGS := -I./include -I./include/type -Wall -Wextra -Werror

# Directories
SRCDIR := src
INCDIR := include
OBJDIR := obj

# Create a list of source files
SRC := $(wildcard $(SRCDIR)/**/*.c) $(wildcard $(SRCDIR)/*.c)

# Create a list of object files
OBJ := $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRC))

# Target executable
TARGET := main

# Default target
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

# Compile source files into object files
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean up build artifacts
clean:
	rm -rf $(OBJDIR) $(TARGET)

# PHONY targets
.PHONY: all clean
