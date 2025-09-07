#include <iostream>
#include <chrono>
#include <queue>
#include <vector>
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

typedef struct Vertice_DP {
    bool visitado;
    int peso;
    bool aresta;
} Vertice_DP;

void IniciarGrafo_DP(Vertice_DP **grafo, int vert){
    for(int i=0;i<vert;i++){
        grafo[i] = new Vertice_DP[vert];
        for(int j=0;j<vert;j++){
            grafo[i][j].visitado = false;
            grafo[i][j].peso = 0;
            grafo[i][j].aresta = false;
        }
    }
}

void CriarArestas_DP(Vertice_DP **grafo, int vert){
    cout << "Digite os dois Vertice_DPs que vao ser conectados e o seu peso (Digite 0 0 0 para sair)." << endl;
    int v1,v2;
    int peso;
    int maxvert = (vert*(vert-1))/2;
    while (maxvert > 0)
    {
        cin >> v1 >> v2 >> peso;

        if (v1 == 0 && v2 == 0 && peso == 0)
        {
            break;
        }

        if (v1 == v2)
        {
            cout << "Loops nao existem em um grafo simples! ~Por favor escreva uma aresta valida." << endl;
            continue;
        }

        if (v1 < 0 || v1 >= vert || v2 < 0 || v2 >= vert)
        {
            cout << "Vertice_DPs invalidas! Por favor escreva arestas entre 0 e " << vert - 1 << endl;
            continue;
        }

        grafo[v1][v2].aresta = true;
        grafo[v1][v2].peso = peso;
        maxvert -= 1;
    }
}

void BFS_DP(Vertice_DP **grafo, int vert, int origem, int destino) {
    vector<int> distancia(vert, -1);
    vector<int> anterior(vert, -1);
    queue<int> fila;

    distancia[origem] = 0;
    fila.push(origem);

    while (!fila.empty()) {
        int atual = fila.front();
        fila.pop();

        for (int i = 0; i < vert; i++) {
            if (grafo[atual][i].aresta && distancia[i] == -1) {
                distancia[i] = distancia[atual] + 1;
                anterior[i] = atual;
                fila.push(i);
                if (i == destino) break;
            }
        }
    }

    if (distancia[destino] == -1) {
        cout << "Nao existe caminho entre " << origem << " e " << destino << endl;
    } else {
        cout << "Menor caminho tem " << distancia[destino] << " arestas." << endl;
        vector<int> caminho;
        for (int v = destino; v != -1; v = anterior[v])
            caminho.push_back(v);
        cout << "Caminho: ";
        for (int i = caminho.size() - 1; i >= 0; i--)
            cout << caminho[i] << (i ? " -> " : "\n");
    }
}

void Dijkstra_DP(Vertice_DP **grafo, int vert, int origem, int destino) {
    vector<int> dist(vert, numeric_limits<int>::max());
    vector<int> anterior(vert, -1);
    vector<bool> visitado(vert, false);

    dist[origem] = 0;

    for (int count = 0; count < vert; count++) {
        int u = -1;
        int minDist = numeric_limits<int>::max();
        for (int i = 0; i < vert; i++) {
            if (!visitado[i] && dist[i] < minDist) {
                minDist = dist[i];
                u = i;
            }
        }
        if (u == -1) break;

        visitado[u] = true;

        for (int v = 0; v < vert; v++) {
            if (grafo[u][v].aresta && !visitado[v]) {
                int peso = grafo[u][v].peso;
                if (dist[u] + peso < dist[v]) {
                    dist[v] = dist[u] + peso;
                    anterior[v] = u;
                }
            }
        }
    }

    if (dist[destino] == numeric_limits<int>::max()) {
        cout << "Nao existe caminho entre " << origem << " e " << destino << endl;
    } else {
        cout << "Menor caminho tem custo " << dist[destino] << "." << endl;
        vector<int> caminho;
        for (int v = destino; v != -1; v = anterior[v])
            caminho.push_back(v);
        cout << "Caminho: ";
        for (int i = caminho.size() - 1; i >= 0; i--)
            cout << caminho[i] << (i ? " -> " : "\n");
    }
}

void ImprimirGrafo_DP(Vertice_DP **grafo, int vert){
    char letra = 'A';   
    cout << "  ";
    for (int i = 0; i < vert; i++) {
        cout << letra << "      ";
        letra += 1;     
    }
    cout << endl;
    letra = 'A';
    for(int i=0;i<vert;i++){
        cout << letra << " ";
        for(int j=0;j<vert;j++){
            cout << grafo[i][j].aresta << " P:" << grafo[i][j].peso << "| ";
        }
        cout << endl;
        letra += 1;
    }
}

void DFS_DP(Vertice_DP **grafo, int vert, int atual) {
    grafo[atual][atual].visitado = true;
    for (int i = 0; i < vert; i++) {
        if (grafo[atual][i].aresta && !grafo[i][i].visitado) {
            DFS_DP(grafo, vert, i);
        }
    }
}

int main() {
    int qntVertices;
    cin >> qntVertices;

    Vertice_DP **grafo = new Vertice_DP*[qntVertices];
    IniciarGrafo_DP(grafo, qntVertices);

    CriarArestas_DP(grafo, qntVertices);

    int origem, destino;
    cout << "Digite o vertice de origem e destino para o Dijkstra: ";
    cin >> origem >> destino;

    auto inicio = high_resolution_clock::now();
    Dijkstra_DP(grafo, qntVertices, origem, destino);
    auto fim = high_resolution_clock::now();
    auto duracao_ms = duration_cast<milliseconds>(fim - inicio);
    cout << duracao_ms.count();
}