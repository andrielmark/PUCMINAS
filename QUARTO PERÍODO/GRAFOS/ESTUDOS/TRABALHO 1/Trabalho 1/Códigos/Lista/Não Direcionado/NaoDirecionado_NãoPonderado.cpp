// Trabalho_1/NãoDirecionado_NãoPonderado.cpp

#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <limits> // Necessário para numeric_limits
#include <algorithm> // Necessário para reverse

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;

struct Grafo {
    int n; // número de vértices
    /*
       Grafo não direcionado e não ponderado usando lista de adjacência:

       1. Cada aresta (u, v) é armazenada em adj[u] e adj[v], 
          pois a conexão é bidirecional.
       2. Eficiência de memória: lista de adjacência ocupa O(n + m), ideal para grafos esparsos.
       3. Fácil acesso aos vizinhos de cada vértice.
       4. Ideal para algoritmos de busca, componentes conectados, etc.
    */
    vector<vector<int>> adj; // lista de adjacência

    Grafo(int n) {
        this->n = n;
        adj.resize(n);
    }

    void addAresta(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // adiciona a aresta no sentido contrário
    }

    void print() {
        for (int i = 0; i < n; i++) {
            cout << i << ": ";
            for (int v : adj[i]) {
                cout << v << " ";
            }
            cout << "\n";
        }
    }

    // Função para fazer a busca com profundidade em grafo
    void DFS(int inicio) {
        vector<bool> visitado(n, false);
        DFS_Recursivo(inicio, visitado);
        cout << "\n";
    }

    // Função auxiliar para fazer a busca com profundidade em grafo
    void DFS_Recursivo(int atual, vector<bool>& visitado) {
        visitado[atual] = true;
        for (int vizinho : adj[atual]) {
            if (!visitado[vizinho]) {
                DFS_Recursivo(vizinho, visitado);
            }
        }
    }

    // Função para fazer a busca com largura em grafo
    void BFS(int inicio) {
        vector<bool> visitado(n, false);
        queue<int> q;

        q.push(inicio);
        visitado[inicio] = true;

        while (!q.empty()) {
            int atual = q.front();
            q.pop();

            for (int vizinho : adj[atual]) {
                if (!visitado[vizinho]) {
                    visitado[vizinho] = true;
                    q.push(vizinho);
                }
            }
        }
        cout << "\n";
    }
};

int main()
{
    int qntVertices;
    cin >> qntVertices;
    Grafo g(qntVertices);

    int v1, v2;
    while (1) {
        cin >> v1;
        if (v1 == -1) {
            break;
        }
        cin >> v2;
        g.addAresta(v1, v2);
    }

    cout << "\nGrafo carregado com sucesso!\n";

    cout << "\nBusca em Profundidade (DFS) a partir do vertice 0:\n";
    auto inicio_dfs = high_resolution_clock::now();
    g.DFS(0);
    auto fim_dfs = high_resolution_clock::now();
    auto duracao_dfs = duration_cast<nanoseconds>(fim_dfs - inicio_dfs);
    cout << "Tempo do DFS: " << duracao_dfs.count() << " ms" << endl;

    cout << "\nBusca em Largura (BFS) a partir do vertice 0:\n";
    auto inicio_bfs = high_resolution_clock::now();
    g.BFS(0);
    auto fim_bfs = high_resolution_clock::now();
    auto duracao_bfs = duration_cast<nanoseconds>(fim_bfs - inicio_bfs);
    cout << "Tempo do BFS: " << duracao_bfs.count() << " ms" << endl;

    return 0;
}