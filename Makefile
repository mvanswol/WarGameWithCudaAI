CUDA_INSTALL_PATH := /usr/local/cuda

CXX := g++
LINK := g++ -fPIC
NVCC  := nvcc

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS) -m64 -arch=compute_20
CXXFLAGS += $(COMMONFLAGS) -m64
CFLAGS += $(COMMONFLAGS)

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart
OBJS = benchmarkGame.cpp.o board.cpp.o cudaGameTree.cu.o gameTree.cpp.o gpuPlayer.cpp.o move.cpp.o parallelGameTree.cpp.o playerAI.cpp.o gameTreeNode.cpp.o
TARGET = benchmarkGame
LINKLINE = $(LINK) -m64 -o $(TARGET) $(OBJS) $(LIB_CUDA)


.SUFFIXES: .c .cpp .cu .o


%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(LINKLINE)

clean:
	rm -f *.o $(TARGET)
