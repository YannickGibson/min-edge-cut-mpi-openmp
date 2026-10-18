# Minimum edge cut · MPI + OpenMP

C++17 branch-and-bound search for a minimum-weight cut with prescribed partition sizes `a` and `n − a` in an undirected, nonnegative weighted graph.

Four implementations compare sequential DFS, OpenMP recursive tasks, OpenMP dynamic scheduling over BFS-generated subtrees, and hybrid MPI + OpenMP. The hybrid solver dynamically distributes subtrees across MPI workers, shares the incumbent bound with new work, and searches each subtree using OpenMP tasks.

Requires a C++17 compiler, OpenMP, and MPI (e.g. Open MPI).

```sh
make -j
./bin/seq
OMP_NUM_THREADS=4 ./bin/openmp_task
OMP_NUM_THREADS=4 ./bin/openmp_data
OMP_NUM_THREADS=4 mpirun -np 3 ./bin/mpi_openmp_data
```

Run from the repository root. `config.txt` contains the partition size and graph path; the default is a quick 10-vertex example. Included graphs contain a vertex count followed by a weighted adjacency matrix; retain the `graf_<vertices>_<degree>.txt` filename convention. The hybrid run needs at least two ranks, with one coordinator. Solvers print the minimum weight, cut edge count, and elapsed search time; tied optima are not exhaustively enumerated. Exact search has exponential worst-case cost.
