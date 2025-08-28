// Trabalho_1/NãoDirecionado_NãoPonderado.cpp

#include <bits/stdc++.h>
using namespace std;

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
};

int main() {
    Grafo g(5);
    g.addAresta(0, 1);
    g.addAresta(0, 2);
    g.addAresta(1, 3);
    g.addAresta(2, 3);
    g.addAresta(3, 4);

    g.print();
}
