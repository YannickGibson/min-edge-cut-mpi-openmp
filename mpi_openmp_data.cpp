#include <iostream>
#include <cstdlib>
#include <atomic>
#include <vector>
#include <limits>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <chrono>
#include <queue>
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

using namespace std;

const int INF = numeric_limits<int>::max();

// Globální proměnné:
int n, a, k;                    // počet uzlů a velikost menší podmnožiny (cílová)
vector<vector<int>> graph;      // matice sousednosti
int bestCost = INF;             // dosud nalezená minimální váha řezu
// Budeme uchovávat všechny řešení s minimální váhou řezu:
struct Solution {
    vector<int> assignment; // přiřazení: 0 znamená X, 1 znamená Y
    int cost;               // celková váha řezu
    int edgeCount;          // počet hran, které prochází mezi X a Y
};
vector<Solution> bestSolutions;
std::atomic<unsigned long long> recursionCalls{0};

// MPI tags
const int TAG_TASK = 1;
const int TAG_RESULT = 2;
const int TAG_TERMINATE = 3;
int taskDepth = 0; // OpenMP task cutoff within each MPI work item.

// Structure to hold a partial assignment state for BFS
struct AssignmentState {
    vector<int> sol;       // Current assignment
    int idx;               // Current index
    int countX;            // Number of nodes in X
    int countY;            // Number of nodes in Y
    int currentCost;       // Current cost
    int currentEdgeCount;  // Current edge count
};

// Pro výpočet dolního odhadu dodatečné váhy řezu pro dosud nepřiřazené uzly.
// Pro každý nepřiřazený uzel spočítáme, o kolik by se řeč zvýšila, pokud by byl
// přiřazen do X (tj. všechny hrany z tohoto uzlu do uzlů, které jsou již v Y)
// nebo do Y (tj. hrany do uzlů v X). Jako dolní odhad vezmeme minimum z těchto dvou.
int computeLowerBound(const vector<int>& sol, int idx, int currentCost) {
    int lb = 0;
    // Pro každý uzel, který ještě nebyl přiřazen (od idx do n-1)
    for (int i = idx; i < n; i++) {
        int costIfX = 0; // kdyby byl přiřazen do X (značka 0)
        int costIfY = 0; // kdyby byl přiřazen do Y (značka 1)
        // Projdeme všechny uzly, které už mají přiřazenou hodnotu
        for (int j = 0; j < idx; j++) {
            if (sol[j] == -1) continue; // mělo by nastat jen u nepřiřazených, ale j je vždy přiřazen
            if (sol[j] == 0) {
                // Pokud by i přišel do X, hrana (i,j) by nebyla řezová;
                // kdyby přišel do Y, hrana (i,j) by byla v řezu.
                costIfY += graph[i][j];
            } else { // sol[j] == 1
                // Kdyby byl i v X, hrana by byla řezová; v Y by nebyla.
                costIfX += graph[i][j];
            }
        }
        lb += min(costIfX, costIfY);
    }
    return currentCost + lb;
}

// Sequential DFS for deeper levels after BFS
void dfs_seq(int idx, vector<int>& sol, int countX, int countY, int currentCost, int currentEdgeCount, std::atomic<int>& localBestCost, vector<Solution>& localBestSolutions) {
    recursionCalls++;

    // Když jsou všechny uzly přiřazeny, ověříme, zda máme validní rozklad:
    if (idx == n) {
        if ((countX == a && countY == n - a) || (countY == a && countX == n - a)) {
            #pragma omp critical
            {
                if (currentCost < localBestCost) {
                    localBestCost = currentCost;
                    localBestSolutions.clear();
                    localBestSolutions.push_back({sol, currentCost, currentEdgeCount});
                } else if (currentCost == localBestCost) {
                    localBestSolutions.push_back({sol, currentCost, currentEdgeCount});
                }
            }
        }
        return;
    }

    // Pruning - check bound
    int lb = computeLowerBound(sol, idx, currentCost);
    if (lb >= localBestCost)
        return;

    // Feasibility check
    int remaining = n - idx;
    if (countX < a && countX + remaining < a) return;
    if (countY < a && countY + remaining < a) return;
    if (countX > a && countY > a) return;

    // Option 1: assign current node to X (marked as 0)
    int addCost = 0;
    int addEdgeCount = 0;
    sol[idx] = 0;
    for (int j = 0; j < idx; j++) {
        if (sol[j] != -1 && sol[j] == 1) {
            addCost += graph[idx][j];
            if (graph[idx][j] > 0) addEdgeCount++;
        }
    }
    #pragma omp task firstprivate(sol, idx, countX, countY, currentCost, currentEdgeCount, addCost, addEdgeCount) shared(localBestCost, localBestSolutions) if(idx < taskDepth)
    {
        dfs_seq(idx + 1, sol, countX + 1, countY, currentCost + addCost, currentEdgeCount + addEdgeCount, localBestCost, localBestSolutions);
    }

    // Option 2: assign current node to Y (marked as 1)
    addCost = 0;
    addEdgeCount = 0;
    sol[idx] = 1;
    for (int j = 0; j < idx; j++) {
        if (sol[j] != -1 && sol[j] == 0) {
            addCost += graph[idx][j];
            if (graph[idx][j] > 0) addEdgeCount++;
        }
    }
    #pragma omp task firstprivate(sol, idx, countX, countY, currentCost, currentEdgeCount, addCost, addEdgeCount) shared(localBestCost, localBestSolutions) if(idx < taskDepth)
    {
        dfs_seq(idx + 1, sol, countX, countY + 1, currentCost + addCost, currentEdgeCount + addEdgeCount, localBestCost, localBestSolutions);
    }

    #pragma omp taskwait

    // Reset current node
    sol[idx] = -1;
}

// BFS to generate initial states for parallel processing
vector<AssignmentState> generateInitialStates(int bfsDepth) {
    vector<AssignmentState> states;
    queue<AssignmentState> q;

    // Start with initial state
    vector<int> initialSol(n, -1);
    q.push({initialSol, 0, 0, 0, 0, 0});

    while (!q.empty()) {
        AssignmentState current = q.front();
        q.pop();

        // If we've reached the desired depth, add to states for parallel processing
        if (current.idx == bfsDepth) {
            states.push_back(current);
            continue;
        }

        // Check bounds and feasibility
        int lb = computeLowerBound(current.sol, current.idx, current.currentCost);
        if (lb >= bestCost) continue;

        int remaining = n - current.idx;
        if (current.countX < a && current.countX + remaining < a) continue;
        if (current.countY < a && current.countY + remaining < a) continue;
        if (current.countX > a && current.countY > a) continue;

        // Option 1: assign to X
        vector<int> solX = current.sol;
        solX[current.idx] = 0;
        int addCostX = 0;
        int addEdgeCountX = 0;

        for (int j = 0; j < current.idx; j++) {
            if (solX[j] != -1 && solX[j] == 1) {
                addCostX += graph[current.idx][j];
                if (graph[current.idx][j] > 0) addEdgeCountX++;
            }
        }

        q.push({solX, current.idx + 1, current.countX + 1, current.countY,
                current.currentCost + addCostX, current.currentEdgeCount + addEdgeCountX});

        // Option 2: assign to Y
        vector<int> solY = current.sol;
        solY[current.idx] = 1;
        int addCostY = 0;
        int addEdgeCountY = 0;

        for (int j = 0; j < current.idx; j++) {
            if (solY[j] != -1 && solY[j] == 0) {
                addCostY += graph[current.idx][j];
                if (graph[current.idx][j] > 0) addEdgeCountY++;
            }
        }

        q.push({solY, current.idx + 1, current.countX, current.countY + 1,
                current.currentCost + addCostY, current.currentEdgeCount + addEdgeCountY});
    }

    return states;
}

// Serialize AssignmentState for MPI
void serializeState(const AssignmentState& state, vector<int>& buffer) {
    int solSize = state.sol.size();
    buffer.resize(5 + solSize);

    // Copy solution vector
    for (int i = 0; i < solSize; i++) {
        buffer[i] = state.sol[i];
    }

    // Copy remaining fields
    buffer[solSize] = state.idx;
    buffer[solSize + 1] = state.countX;
    buffer[solSize + 2] = state.countY;
    buffer[solSize + 3] = state.currentCost;
    buffer[solSize + 4] = state.currentEdgeCount;
}

// Deserialize AssignmentState from MPI buffer
AssignmentState deserializeState(const vector<int>& buffer) {
    int solSize = n;
    AssignmentState state;

    // Extract solution vector
    state.sol.resize(solSize);
    for (int i = 0; i < solSize; i++) {
        state.sol[i] = buffer[i];
    }

    // Extract remaining fields
    state.idx = buffer[solSize];
    state.countX = buffer[solSize + 1];
    state.countY = buffer[solSize + 2];
    state.currentCost = buffer[solSize + 3];
    state.currentEdgeCount = buffer[solSize + 4];

    return state;
}

// Serialize Solution for MPI
void serializeSolution(const Solution& solution, vector<int>& buffer) {
    int solSize = solution.assignment.size();
    buffer.resize(2 + solSize);

    // Copy assignment vector
    for (int i = 0; i < solSize; i++) {
        buffer[i] = solution.assignment[i];
    }

    // Copy remaining fields
    buffer[solSize] = solution.cost;
    buffer[solSize + 1] = solution.edgeCount;
}

// Deserialize Solution from MPI buffer
Solution deserializeSolution(const vector<int>& buffer) {
    int solSize = n;
    Solution solution;

    // Extract assignment vector
    solution.assignment.resize(solSize);
    for (int i = 0; i < solSize; i++) {
        solution.assignment[i] = buffer[i];
    }

    // Extract remaining fields
    solution.cost = buffer[solSize];
    solution.edgeCount = buffer[solSize + 1];

    return solution;
}

void load_k(const string& filename) {
    // a - |X|, k - průmerný stupeň uzlu

    size_t start = filename.find('/') + 1;
    start = filename.find('_', start) + 1;
    start = filename.find('_', start);
    size_t end = filename.find('.', start);
    size_t endb = filename.find('b', start);
    size_t minEnd = std::min(end, endb);

    if (start == string::npos || minEnd == string::npos) {
        cerr << "Invalid filename format!" << endl;
        exit(1);
    }

    k = stoi(filename.substr(start + 1, minEnd - start - 1));
}

void loadGraph(const string& filename) {
    ifstream file(filename);
    if (!file) {
        cerr << "Error opening file!" << endl;
        exit(1);
    }

    if (!(file >> n) || n < 2 || a >= n) {
        cerr << "Invalid graph size or partition size." << endl;
        exit(1);
    }
    a = min(a, n - a);
    load_k(filename);

    graph.assign(n, vector<int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (!(file >> graph[i][j]) || graph[i][j] < 0) {
                cerr << "Expected a nonnegative weighted adjacency matrix." << endl;
                exit(1);
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (graph[i][j] != graph[j][i]) {
                cerr << "Expected an undirected graph." << endl;
                exit(1);
            }
        }
    }
    file.close();
}

void printNeigborMatrix() {
    cout << "Matice sousedů:" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << right << setw(5) << graph[i][j] << " ";
        }
        cout << endl;
    }
}

std::pair<int, std::string> loadConfig(const std::string& filename) {
    int a = 0;
    std::string graphFile;
    std::ifstream config(filename);

    if (!(config >> a >> graphFile) || a < 1) {
        std::cerr << "Expected config.txt: <partition size> <graph path>" << std::endl;
        std::exit(1);
    }

    return {a, graphFile};
}

// Function for master process
void masterProcess(int numProcs) {
    auto start = std::chrono::high_resolution_clock::now();

    // Run BFS to generate initial states
    int bfsDepth = min(10, n/2);
    vector<AssignmentState> states = generateInitialStates(bfsDepth);

    cout << "Master: BFS generated " << states.size() << " initial states at depth " << bfsDepth << endl;

    // Distribute tasks to slave processes
    int numTasks = states.size();
    int tasksSent = 0;
    int tasksCompleted = 0;
    // A completed worker receives another subtree and the latest global bound.
    // Workers with no remaining work receive exactly one termination message.
    auto dispatch = [&](int worker) {
        if (tasksSent < numTasks) {
            vector<int> buffer;
            serializeState(states[tasksSent++], buffer);
            buffer.push_back(bestCost);
            MPI_Send(buffer.data(), buffer.size(), MPI_INT, worker, TAG_TASK, MPI_COMM_WORLD);
        } else {
            MPI_Send(nullptr, 0, MPI_INT, worker, TAG_TERMINATE, MPI_COMM_WORLD);
        }
    };
    for (int worker = 1; worker < numProcs; ++worker) dispatch(worker);
    while (tasksCompleted < numTasks) {
        MPI_Status status;
        vector<int> buffer(n + 2);
        MPI_Recv(buffer.data(), buffer.size(), MPI_INT, MPI_ANY_SOURCE,
                 TAG_RESULT, MPI_COMM_WORLD, &status);
        Solution result = deserializeSolution(buffer);
        if (result.cost < bestCost) {
            bestCost = result.cost;
            bestSolutions = {result};
        }
        ++tasksCompleted;
        dispatch(status.MPI_SOURCE);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Output results

    if (bestSolutions.empty()) {
        cout << "No solution found.\n";
    } else {
        cout << "Minimum cut weight: " << bestCost << "\n";
        cout << "Stored optimal candidates: " << bestSolutions.size() << "\n";
        cout << "Cut weight: " << bestSolutions[0].cost << "\n";
        cout << "Cut edges: " << bestSolutions[0].edgeCount << "\n";
    }
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
}

// Function for slave processes
void slaveProcess() {
    while (true) {
        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == TAG_TERMINATE) {
            MPI_Recv(nullptr, 0, MPI_INT, 0, TAG_TERMINATE, MPI_COMM_WORLD, &status);
            break;
        }
        vector<int> buffer(n + 6);
        MPI_Recv(buffer.data(), buffer.size(), MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, &status);
        AssignmentState state = deserializeState(buffer);
        std::atomic<int> localBestCost{buffer.back()};
        vector<Solution> localBestSolutions;
        taskDepth = min(n, state.idx + 3);
        #pragma omp parallel
        {
            #pragma omp single
            dfs_seq(state.idx, state.sol, state.countX, state.countY,
                    state.currentCost, state.currentEdgeCount,
                    localBestCost, localBestSolutions);
        }
        Solution result{vector<int>(n, -1), INF, 0};
        if (!localBestSolutions.empty()) result = localBestSolutions.front();
        serializeSolution(result, buffer);
        MPI_Send(buffer.data(), buffer.size(), MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
    }
}

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Initialize MPI
    int rank, numProcs;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if (numProcs < 2) {
        if (rank == 0) cerr << "Use at least two MPI ranks (one coordinator and workers)." << endl;
        MPI_Finalize();
        return 1;
    }
    if (argc > 1) {
        int threads = atoi(argv[1]);
        if (threads < 1) MPI_Abort(MPI_COMM_WORLD, 1);
        omp_set_num_threads(threads);
    }

    // Load graph (all processes need this)
    auto [new_a, file] = loadConfig("config.txt");
    a = new_a;
    loadGraph(file);

    if (rank == 0) {
        // Master process
        cout << "Graph: " << file << endl;
        cout << "Using " << omp_get_max_threads() << " threads" << endl;
        cout << "n: " << n << ", a: " << a << ", k: " << k << endl;
        cout << "Running with " << numProcs << " MPI processes" << endl;
        masterProcess(numProcs);
    } else {
        // Slave process
        slaveProcess();
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
