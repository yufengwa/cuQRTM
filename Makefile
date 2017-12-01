#! /bin/sh

# compiler
CC=nvcc
# set CUDA and MPI environment path 
INC=-I/usr/local/cuda-8.0/include -I/home/wyf/intel/impi/5.0.1.035/intel64/include 
LIB=-L/usr/local/cuda-8.0/lib64 -L/home/wyf/intel/impi/5.0.1.035/intel64/lib/ 
# set CUDA and MPI Dynamic link library
LINK= -lcudart -lcufft -lm -lmpich -lpthread -lrt -DMPICH_IGNORE_CXX_SEEK  -DMPICH_SKIP_MPICXX

# CUDA and C++ source codes
SOURCES=CUDAQRTM.cu QRTM.cpp
EXECNAME=QRTM

# Execution
all:
	$(CC) -v -o $(EXECNAME) $(SOURCES) $(INC) $(LIB) $(LINK)
	rm -f *.o 
	nohup mpirun -np 1 ./QRTM &