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

// Rekurzivní DFS s branch and bound.
// idx          - aktuální index uzlu, který přiřazujeme (0-indexováno)
// sol          - aktuální částečné přiřazení, kde sol[i] = 0 (X), 1 (Y) nebo -1 (nepřiřazeno)
// countX, countY - počet uzlů již přiřazených do X a Y
// currentCost  - dosavadní váha řezu (součet vah hran mezi uzly s různými přiřazeními)
// currentEdgeCount - počet hran, které již přispívají do řezu
void dfs(int idx, vector<int>& sol, int countX, int countY, int currentCost, int currentEdgeCount) {

    recursionCalls++;
    // Když jsou všechny uzly přiřazeny, ověříme, zda máme validní rozklad:
    // validní je, pokud |X| = a a |Y| = n - a, nebo naopak.
    if (idx == n) {
        if ((countX == a && countY == n - a) || (countY == a && countX == n - a)) {
            // Našli jsme kandidátní řešení, porovnáme váhu řezu s dosud nejlepším.

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

    // Prořezání větve – pokud již nyní překročíme nejlepší nalezené řešení,
    // ukončíme tuto větev.
    int lb;
    lb = computeLowerBound(sol, idx, currentCost);
    if (lb >= bestCost)
        return;

    // Dále provádíme kontrolu feasibility (jestli je ještě možné dosáhnout validního rozkladu):
    int remaining = n - idx;
    // Pokud dosud není ani X ani Y "dost velká", musíme zajistit, aby bylo možné dosáhnout velikosti a.
    if (countX < a && countX + remaining < a) return;
    if (countY < a && countY + remaining < a) return;
    // Jestliže obě množiny již překročily velikost a, nemůžeme dosáhnout požadovaného rozkladu.
    if (countX > a && countY > a) return;

    // Možnost 1: přiřadíme aktuální uzel do X (značíme 0)
    int addCost = 0;
    int addEdgeCount = 0;
        sol[idx] = 0;
        // Přičteme příspěvek z hran mezi aktuálním uzlem a již přiřazenými uzly,
        // které jsou v opačné množině (tj. v Y).
        for (int j = 0; j < idx; j++) {
            if (sol[j] != -1 && sol[j] == 1) {
                addCost += graph[idx][j];
                if (graph[idx][j] > 0) addEdgeCount++;
            }
        }

    # pragma omp task shared(bestCost, bestSolutions, recursionCalls) if(idx <= n - 9)
    {
        dfs(idx + 1, sol, countX + 1, countY, currentCost + addCost, currentEdgeCount + addEdgeCount);
    }

    // Možnost 2: přiřadíme aktuální uzel do Y (značíme 1)
        sol[idx] = 1;
        addCost = 0;
        addEdgeCount = 0;
        for (int j = 0; j < idx; j++) {
            if (sol[j] != -1 && sol[j] == 0) {
                addCost += graph[idx][j];
                if (graph[idx][j] > 0) addEdgeCount++;
            }
        }

    # pragma omp task shared(bestCost, bestSolutions, recursionCalls) if(idx <= n - 9)
    {
        dfs(idx + 1, sol, countX, countY + 1, currentCost + addCost, currentEdgeCount + addEdgeCount);
    }
    # pragma omp taskwait

    // Zpětné nastavení – odznačíme aktuální uzel
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
    cout << "n: " << n << ", a: " << a << ", k: " << k << endl;
    cout << "Graph: " << file << endl;

    // Inicializace částečného přiřazení: -1 znamená, že uzel ještě není přiřazen.
    vector<int> sol(n, -1);

    auto start = std::chrono::high_resolution_clock::now();
    // Spustíme DFS s branch and bound.
    #pragma omp parallel
    {
        #pragma omp single
        {
            dfs(0, sol, 0, 0, 0, 0);
        }
    }

    // Calculate the duration -> Output
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    // Výpis výsledků.
    if (bestSolutions.empty()) {
        cout << "No solution found.\n";
    } else {
        cout << "Minimum cut weight: " << bestCost << "\n";
        cout << "Recursive calls: " << recursionCalls << endl;
        cout << "Stored optimal candidates: " << bestSolutions.size() << "\n";
        cout << "Cut weight: " << bestSolutions[0].cost << "\n\n";
        cout << "\nCut edges: " << bestSolutions[0].edgeCount << "\n";
    }

    return 0;
}
