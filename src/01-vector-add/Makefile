# CUDA Makefile for Vector Addition Project
# Author: Generated for triton_kernel project

# Compiler and flags
NVCC = nvcc
CXX = g++

# Target architecture (adjust based on your GPU)
# Common architectures:
# sm_50: Maxwell (GTX 900 series)
# sm_60: Pascal (GTX 10 series) 
# sm_70: Volta (V100)
# sm_75: Turing (RTX 20 series)
# sm_80: Ampere (A100, RTX 30 series)
# sm_86: Ampere (RTX 30 series)
# sm_89: Ada Lovelace (RTX 40 series)
# sm_90: Hopper (H100, H20) - requires CUDA 12+
# For CUDA 11.5, use sm_87 for H20 compatibility
ARCH = sm_87

# CUDA flags
NVCC_FLAGS = -arch=$(ARCH) -std=c++14
NVCC_DEBUG_FLAGS = -g -G -O0 -arch=$(ARCH) -std=c++14
NVCC_RELEASE_FLAGS = -O3 -arch=$(ARCH) -std=c++14 -DNDEBUG

# Include and library paths
CUDA_PATH = /usr/local/cuda
INCLUDES = -I$(CUDA_PATH)/include
LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lm

# Source and target files
SOURCES = vector_add.cu
TARGET = vector_add
DEBUG_TARGET = vector_add_debug
RELEASE_TARGET = vector_add_release

# Default target
all: $(TARGET)

# Basic compilation
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# Debug build with debugging symbols
debug: $(DEBUG_TARGET)
$(DEBUG_TARGET): $(SOURCES)
	$(NVCC) $(NVCC_DEBUG_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# Release build with optimizations
release: $(RELEASE_TARGET)
$(RELEASE_TARGET): $(SOURCES)
	$(NVCC) $(NVCC_RELEASE_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Run debug version
run-debug: debug
	cuda-gdb ./$(DEBUG_TARGET)

# Profile with nvprof (deprecated but might still work)
profile: $(TARGET)
	nvprof ./$(TARGET)

# Profile with nsys (modern profiler)
profile-nsys: $(TARGET)
	nsys profile --trace=cuda,osrt ./$(TARGET)

# Compile with verbose output
verbose: $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -v -o $(TARGET) $< $(LIBS)

# Compile and show PTX assembly
ptx: $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) --ptx -o vector_add.ptx $<

# Compile and show SASS assembly  
sass: $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) --cubin -o vector_add.cubin $<
	cuobjdump --dump-sass vector_add.cubin

# Check CUDA installation
check-cuda:
	@echo "CUDA Version:"
	nvcc --version
	@echo "GPU Information:"
	nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits

# Clean build artifacts
clean:
	rm -f $(TARGET) $(DEBUG_TARGET) $(RELEASE_TARGET)
	rm -f *.ptx *.cubin
	rm -f *.o *.so
	rm -f *.nsys-rep *.sqlite

# Create backup
backup:
	tar -czf vector_add_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz *.cu *.py Makefile

# Show help
help:
	@echo "Available targets:"
	@echo "  all         - Build basic version (default)"
	@echo "  debug       - Build debug version with symbols"
	@echo "  release     - Build optimized release version"
	@echo "  run         - Build and run the program"
	@echo "  run-debug   - Build and run with cuda-gdb"
	@echo "  profile     - Build and profile with nvprof"
	@echo "  profile-nsys- Build and profile with nsys"
	@echo "  verbose     - Build with verbose output"
	@echo "  ptx         - Generate PTX assembly"
	@echo "  sass        - Generate SASS assembly"
	@echo "  check-cuda  - Check CUDA installation"
	@echo "  clean       - Remove build artifacts"
	@echo "  backup      - Create backup archive"
	@echo "  help        - Show this help message"

# Phony targets
.PHONY: all debug release run run-debug profile profile-nsys verbose ptx sass check-cuda clean backup help
