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

using namespace std;

const int INF = numeric_limits<int>::max();

// Globální proměnné:
int n, a, k;                        // počet uzlů a velikost menší podmnožiny (cílová)
vector<vector<int>> graph;       // matice sousednosti
std::atomic<int> bestCost{INF};              // dosud nalezená minimální váha řezu
// Budeme uchovávat všechny řešení s minimální váhou řezu:
struct Solution {
    vector<int> assignment; // přiřazení: 0 znamená X, 1 znamená Y
    int cost;               // celková váha řezu
    int edgeCount;          // počet hran, které prochází mezi X a Y
};
vector<Solution> bestSolutions;
std::atomic<unsigned long long> recursionCalls{0};


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

// Structure to hold a partial assignment state for BFS
struct AssignmentState {
    vector<int> sol;       // Current assignment
    int idx;               // Current index
    int countX;            // Number of nodes in X
    int countY;            // Number of nodes in Y
    int currentCost;       // Current cost
    int currentEdgeCount;  // Current edge count
};

// Sequential DFS for deeper levels after BFS
void dfs_seq(int idx, vector<int>& sol, int countX, int countY, int currentCost, int currentEdgeCount) {
    recursionCalls++;

    // Když jsou všechny uzly přiřazeny, ověříme, zda máme validní rozklad:
    if (idx == n) {
        if ((countX == a && countY == n - a) || (countY == a && countX == n - a)) {
            #pragma omp critical
            {
                if (currentCost < bestCost) {
                    bestCost = currentCost;
                    bestSolutions.clear();
                    bestSolutions.push_back({sol, currentCost, currentEdgeCount});
                } else if (currentCost == bestCost) {
                    bestSolutions.push_back({sol, currentCost, currentEdgeCount});
                }
            }
        }
        return;
    }

    // Pruning - check bound
    int lb = computeLowerBound(sol, idx, currentCost);
    if (lb >= bestCost)
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
    dfs_seq(idx + 1, sol, countX + 1, countY, currentCost + addCost, currentEdgeCount + addEdgeCount);

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
    dfs_seq(idx + 1, sol, countX, countY + 1, currentCost + addCost, currentEdgeCount + addEdgeCount);

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

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);




    // Načteme vstup:
    auto [new_a, file] = loadConfig("config.txt");
    a = new_a;
    loadGraph(file);
    cout << "Graph: " << file << endl;
    cout << "Using " << omp_get_max_threads() << " threads" << endl;
    cout << "n: " << n << ", a: " << a << ", k: " << k << endl;

    auto start = std::chrono::high_resolution_clock::now();

    // Run BFS to generate initial states (typically to depth 4-6 depending on graph size)
    int bfsDepth = min(6, n/2); // Choose appropriate depth
    vector<AssignmentState> states = generateInitialStates(bfsDepth);

    cout << "BFS generated " << states.size() << " initial states at depth " << bfsDepth << endl;

    // Process states in parallel
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < states.size(); i++) {
            AssignmentState& state = states[i];
            vector<int> sol = state.sol;
            dfs_seq(state.idx, sol, state.countX, state.countY,
                    state.currentCost, state.currentEdgeCount);
        }
    }

    // Calculate the duration -> Output
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Výpis výsledků.
    if (bestSolutions.empty()) {
        cout << "No solution found.\n";
    } else {
        cout << "Minimum cut weight: " << bestCost << "\n";
        cout << "Recursive calls: " << recursionCalls << endl;
        cout << "Stored optimal candidates: " << bestSolutions.size() << "\n";
        cout << "Cut weight: " << bestSolutions[0].cost << "\n";
        cout << "Cut edges: " << bestSolutions[0].edgeCount << "\n";
    }
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}
