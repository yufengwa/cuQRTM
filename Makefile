#! /bin/sh

CC=nvcc
CFLAGS=-I/usr/local/cuda-7.5/include -I/usr/local/include 
LDFLAGS=-L/usr/local/cuda-7.5/lib64 -L/usr/local/lib/
LIB= -lcudart -lcufft -lm -lmpich -lpthread -lrt
MPICH_FLAG= -DMPICH_IGNORE_CXX_SEEK  -DMPICH_SKIP_MPICXX
SOURCES=CUDAQRTM.cu QRTM.cpp
EXECNAME=QRTM
all:
	$(CC) -v -o $(EXECNAME) $(SOURCES) $(LIB) $(LDFLAGS) $(CFLAGS) $(MPICH_FLAG)
	rm -f *.o 
