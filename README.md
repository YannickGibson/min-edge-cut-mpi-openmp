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

## Results

The original April 2025 experiments achieved **9.35× OpenMP speedup**, reducing search time from **631.73 s to 67.55 s** on `graf_32_25.txt` (32 vertices, partition size `a = 12`). Speedup is relative to the sequential baseline on the same input.

| Implementation | Configuration | Time (s) | Speedup |
|---|---|---:|---:|
| Sequential | 1 thread | 631.73 | 1.00× |
| OpenMP work sharing | 6 threads | 154.25 | 4.10× |
| OpenMP work sharing | 12 threads | 106.12 | 5.95× |
| OpenMP work sharing | 48 threads | 67.55 | 9.35× |
| Original MPI implementation | 4 ranks, 6 threads configured per rank | 216.89 | 2.91× |

OpenMP scaling showed diminishing returns: efficiency fell from about 68% at 6 threads to 19% at 48. The original MPI implementation gained little from additional configured threads; inspection found that each worker searched on only one OpenMP thread. The current version distributes that search across OpenMP tasks.

These are historical report measurements, not benchmarks of the current code, which includes synchronization and MPI worker fixes. Exact hardware, compiler flags, and repetition methodology are not established by the retained results, so fresh measurements are needed for a reproducible comparison. The current solvers passed 100 runs against brute-force answers on small graphs, covering 1 and 4 threads and 2 and 4 MPI ranks.
