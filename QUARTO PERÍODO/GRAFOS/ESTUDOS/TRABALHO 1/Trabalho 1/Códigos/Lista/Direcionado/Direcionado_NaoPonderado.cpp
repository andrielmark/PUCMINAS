// Trabalho_1/Direcionado_NaoPonderado.cpp

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
    /* Foi escolhida a lista de adjacência para representar este grafo direcionado e não ponderado 
       pelos seguintes motivos:

       1. Eficiência de memória:
          - Diferente da matriz de adjacência (que ocupa O(n²) posições), a lista de adjacência 
            ocupa O(n + m), onde n é o número de vértices e m o número de arestas.
          - Isso é especialmente vantajoso em grafos esparsos (poucas arestas em relação a n²).

       2. Simplicidade para grafos não ponderados:
          - Como não há pesos, basta armazenar o destino de cada aresta.
          - Não é necessário guardar informações extras como valores ou matrizes de custos.

       3. Facilidade de percorrer vizinhos:
          - Ao percorrer a lista de adjacência de um vértice, acessamos apenas seus vizinhos diretos, 
            evitando iterações desnecessárias sobre vértices não conectados (como ocorreria em uma matriz).

       4. Adequação para algoritmos de grafos:
          - Algoritmos de busca (BFS e DFS) e de verificação de conectividade funcionam de forma 
            mais natural e eficiente sobre listas de adjacência.
          - Também facilita operações como adicionar uma aresta, que é O(1) em média.

       Por esses motivos, a lista de adjacência é a escolha mais adequada para representar este 
       grafo direcionado e não ponderado.
    */
    vector<vector<int>> adj; // lista de adjacência

    Grafo(int n) {
        this->n = n;
        adj.resize(n);
    }

    void addAresta(int u, int v) {
        adj[u].push_back(v);
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