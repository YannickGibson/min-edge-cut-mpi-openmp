CXX ?= g++
MPICXX ?= mpic++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
OMPFLAGS ?= -fopenmp

.PHONY: all clean
all: bin/seq bin/openmp_task bin/openmp_data bin/mpi_openmp_data

bin:
	mkdir -p $@

bin/seq: seq.cpp | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/openmp_%: openmp_%.cpp | bin
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

bin/mpi_openmp_data: mpi_openmp_data.cpp | bin
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

clean:
	rm -rf bin
